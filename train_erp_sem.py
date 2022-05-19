from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_full_backward_hook
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from metrics import *
from tqdm import tqdm
from dataset_sem import Dataset
import cv2
import supervision as L
import spherical as S360
from util import load_partial_model
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
import scipy.io
#from model_spherical import Network
from model.spherical_model import spherical_fusion
#from model.spherical_fusion import *
from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils 
from iou import evaluate

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/media/quadro/DATA1/stanford2d3d',
#parser.add_argument('--input_dir', default='/home/quadro/Matterport3d/pano/',
#parser.add_argument('--input_dir', default='/media/rtx2/DATA/Structured3D/',
                    help='input data directory')
parser.add_argument('--trainfile', default='./filenames/train_stanford2d3d.txt',
                    help='train file name')
parser.add_argument('--testfile', default='./filenames/test_stanford2d3d.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=8,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--patchsize', type=list, default=(256, 256),
                    help='patch size')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--nrows', type=int, default=4,
                    help='nrows, options are 4, 6')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='checkpoints',
                    help='save checkpoint path')
parser.add_argument('--save_path', default='./stanford_sem/512x1024/resnet34/visualize_transformer_point_1iter',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
else:
    shutil.rmtree(args.save_path)    
if not os.path.isdir(os.path.join(args.save_path, args.save_checkpoint)):
    os.makedirs(os.path.join(args.save_path, args.save_checkpoint))
# result visualize Path -----------------------
writer_path = os.path.join(args.save_path,args.tensorboard_path)
if not os.path.isdir(writer_path):
    os.makedirs(writer_path)
writer = SummaryWriter(log_dir=writer_path) 

result_view_dir = args.save_path  
shutil.copy('train_erp_depth.py', result_view_dir)
shutil.copy('model/spherical_model_iterative.py', result_view_dir)
shutil.copy('model/spherical_model.py', result_view_dir)
#if os.path.exists('grid'):
#    shutil.rmtree('grid')
#-----------------------------------------

# Random Seed -----------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#------------------------------------------tensorboard_pathf training files
input_dir = args.input_dir
train_file_list = args.trainfile
val_file_list = args.testfile # File with list of validation files
#------------------------------------
#-------------------------------------------------------------------
batch_size = args.batch
visualize_interval = args.visualize_interval
init_lr = 1e-4
fov = (args.fov, args.fov)#(48, 48)
patch_size = args.patchsize
nrows = args.nrows
#-------------------------------------------------------------------
#data loaders
train_dataloader = torch.utils.data.DataLoader(
	dataset=Dataset(
        rotate=True, 
        flip=True,
		root_path=input_dir, 
		path_to_img_list=train_file_list),
	batch_size=batch_size,
	shuffle=True,
	num_workers=8,
	drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
	dataset=Dataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=2,
	shuffle=False,
	num_workers=8,
	drop_last=False)


#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()
network = spherical_fusion()
network = convert_model(network)

# parallel on multi gpu
network = nn.DataParallel(network)
network.cuda()

#----------------------------------------------------------

print('## Batch size: {}'.format(batch_size))
print('## learning rate: {}'.format(init_lr))  
print('## patch size:', patch_size) 
print('## fov:', args.fov)
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in network.parameters() if p.requires_grad is True])))
#--------------------------------------------------

# Optimizer ----------
optimizer = optim.AdamW(list(network.parameters()), 
        lr=init_lr, betas=(0.9, 0.999), weight_decay=0.01)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)   
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.2) 
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6, last_epoch=-1)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
#---------------------


colors = [[0,0,0],
          [0,255,0],
          [0,0,255],
          [0,255,255],
          [255,255,0],
          [255,0,255],
          [100,100,255],
          [200,200,100],
          [170,120,200],
          [255,0,0],
          [200,100,100],
          [10,200,100],
          [200,200,200],
          [50,50,50]]
colors = np.array(colors, dtype=np.uint8)

# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0
    csv_filename = os.path.join(result_view_dir, 'logs/result_log.csv')
    fields = ['epoch', 'iou']
    csvfile = open(csv_filename, 'w', newline='')
    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in range(1, args.epochs+1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        
        #-------------------------------
        network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, _, sem, _) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            bs, _, h, w = rgb.shape
            rgb, sem = rgb.cuda(), sem.cuda()
            mask = sem >= 0       

            equi_outputs = network(rgb, fov, patch_size, nrows)
            loss = F.cross_entropy(equi_outputs, sem, ignore_index=-1)
            
            equi_rgb = rgb.detach().cpu().numpy()
            equi_mask = mask.squeeze(1).detach().cpu().numpy()
            equi_gt = sem.detach().cpu().numpy()
            equi_gt = (equi_gt + 1).astype(np.int32)
            equi_gt *= equi_mask
            sem_prediction = equi_outputs.argmax(1).detach().cpu().numpy()
            sem_prediction = (sem_prediction + 1).astype(np.int32)
            sem_prediction *= equi_mask
            h, w = sem_prediction.shape[-2], sem_prediction.shape[-1]
            if batch_idx % visualize_interval == 0:
                    sem_img = equi_gt[0, :, :]
                    rgb_img = equi_rgb[0, :, :, :].transpose(1, 2, 0)
                    sem_pred_img = sem_prediction[0, :, :]
                    sem_pred_img[rgb_img.sum(-1)==0] = 0
                    sem_img_reshape = np.reshape(colors[sem_img.flatten()], (h, w, 3))
                    sem_pred_img_reshape = np.reshape(colors[sem_pred_img.flatten()], (h, w, 3))
                    cv2.imwrite('{}/equi_rgb_{}.png'.format(result_view_dir, batch_idx), rgb_img*255)
                    cv2.imwrite('{}/equi_pred_{}.png'.format(result_view_dir, batch_idx), sem_pred_img_reshape)
                    cv2.imwrite('{}/equi_gt_{}.png'.format(result_view_dir, batch_idx), sem_img_reshape)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            #scheduler.step()
            total_train_loss += loss.item()
            #total_normal_loss += normal_loss.item()*0.2
            #total_grad_loss += grad_loss.item()*0.05
            global_step += 1
            if batch_idx % visualize_interval == 0 and batch_idx > 0:
                print('[Epoch %d--Iter %d]loss %.4f ' % 
                (epoch, batch_idx, total_train_loss/(batch_idx+1)))


        print('lr for epoch ', epoch, ' ', optimizer.param_groups[0]['lr'])
        torch.save(network.state_dict(), os.path.join(args.save_path, args.save_checkpoint)+'/checkpoint_latest.tar')
        #-----------------------------------------------------------------------------
        scheduler.step()
        # Valid ----------------------------------------------------------------------------------------------------
        if epoch % 2 == 0:
            print('-------------Validate Epoch', epoch, '-----------')
            network.eval()
            gt, pred = [], []
            for batch_idx, (rgb, _, sem, _) in tqdm(enumerate(val_dataloader)):
                bs, _, h, w = rgb.shape
                rgb, sem = rgb.cuda(), sem.cuda()
                mask = sem >= 0

                with torch.no_grad():
                    equi_outputs = network(rgb, fov, patch_size, nrows)
                
                full_mask = mask.squeeze(1)
                full_mask = full_mask.detach().cpu().numpy()
                equi_gt = sem.detach().cpu().numpy()
                gt.append(equi_gt[full_mask].flatten())
                equi_gt = (equi_gt + 1).astype(np.int32)
                equi_rgb = rgb.detach().cpu().numpy()
                sem_prediction = equi_outputs.argmax(1).detach().cpu().numpy()
                pred.append(sem_prediction[full_mask].flatten())
                sem_prediction = (sem_prediction + 1).astype(np.int32)
                h, w = sem_prediction.shape[-2], sem_prediction.shape[-1]
                if batch_idx % visualize_interval == 0:
                    sem_img = equi_gt[0, :, :]
                    rgb_img = equi_rgb[0, :, :, :].transpose(1, 2, 0)
                    sem_pred_img = sem_prediction[0, :, :]
                    sem_pred_img[rgb_img.sum(-1)==0] = 0
                    sem_img_reshape = np.reshape(colors[sem_img.flatten()], (h, w, 3))
                    sem_pred_img_reshape = np.reshape(colors[sem_pred_img.flatten()], (h, w, 3))
                    cv2.imwrite('{}/test_equi_rgb_{}.png'.format(result_view_dir, batch_idx), rgb_img*255)
                    cv2.imwrite('{}/test_equi_pred_{}.png'.format(result_view_dir, batch_idx), sem_pred_img_reshape)
                    cv2.imwrite('{}/test_equi_gt_{}.png'.format(result_view_dir, batch_idx), sem_img_reshape)
                                
            pred = np.hstack(pred)
            gt = np.hstack(gt)
            iou = evaluate(pred, gt)

            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)  
                row = [epoch, '{:.4f}'.format(iou)]
                csvwriter.writerow(row)
    # End Training
    print("Training Ended hahahaha!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    writer.close()
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        