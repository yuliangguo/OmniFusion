from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from metrics import *
from tqdm import tqdm
from dataset_loader_stanford import Dataset
import cv2
import supervision as L
import spherical as S360
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
from model.spherical_model import spherical_fusion
#from model.spherical_fusion import *
from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils 

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/media/rtx2/DATA/stanford2d3d',
#parser.add_argument('--input_dir', default='/home/rtx2/NeurIPS/spherical_mvs/data/omnidepth',
#parser.add_argument('--input_dir', default='/media/rtx2/DATA/Structured3D/',
                    help='input data directory')
parser.add_argument('--trainfile', default='./filenames/train_stanford2d3d.txt',
                    help='train file name')
parser.add_argument('--testfile', default='./filenames/test_stanford2d3d.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch', type=int, default=8,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--patchsize', type=list, default=(128, 128),
                    help='patch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--nrows', type=int, default=4,
                    help='number of rows, options are 3, 4, 5, 6')
parser.add_argument('--confidence', action='store_true', default=True,
                    help='use confidence map or not')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='checkpoints',
                    help='save checkpoint path')
parser.add_argument('--save_path', default='./results/stanford/512x1024/resnet34/visualize_point_1_iter',
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
shutil.copy('model/spherical_model.py', result_view_dir)
#shutil.copy('model/spherical_model_iterative.py', result_view_dir)
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
init_lr = args.lr
fov = (args.fov, args.fov)#(48, 48)
patch_size = args.patchsize
nrows = args.nrows
npatches_dict = {3:10, 4:18, 5:26, 6:46}
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
network = spherical_fusion(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov)
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
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val' : self.val,
            'sum' : self.sum,
            'count' : self.count,
            'avg' : self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']
        
abs_rel_error_meter = AverageMeter()
sq_rel_error_meter = AverageMeter()
lin_rms_sq_error_meter = AverageMeter()
log_rms_sq_error_meter = AverageMeter()
d1_inlier_meter = AverageMeter()
d2_inlier_meter = AverageMeter()
d3_inlier_meter = AverageMeter()

def compute_eval_metrics(output, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output
        gt_depth = gt

        N = depth_mask.sum()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask>0].median() / depth_pred[depth_mask>0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)
        abs_rel_error_meter.update(abs_rel, N)
        sq_rel_error_meter.update(sq_rel, N)
        lin_rms_sq_error_meter.update(rms_sq_lin, N)
        log_rms_sq_error_meter.update(rms_sq_log, N)
        d1_inlier_meter.update(d1, N)
        d2_inlier_meter.update(d2, N)
        d3_inlier_meter.update(d3, N)


# Main Function ---------------------------------------------------------------------------------------------
def main():

    global_step = 0
    global_val = 0
    # save the evaluation results into a csv file
    csv_filename = os.path.join(result_view_dir, 'logs/result_log.csv')
    fields = ['epoch', 'Abs Rel', 'Sq Rel', 'Lin RMSE', 'log RMSE', 'D1', 'D2', 'D3' , 'lr']
    csvfile = open(csv_filename, 'w', newline='')
    with csvfile:
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 

    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    min_error = float("inf")
    for epoch in range(1, args.epochs+1):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        total_depth_loss = 0

        #-------------------------------
        network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, depth, mask) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            bs, _, h, w = rgb.shape
            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()           
            
            equi_outputs = network(rgb)
            
            # error map, clip at 0.1
            error = torch.abs(depth - equi_outputs) * mask
            error[error < 0.1] = 0

            attention_weights = torch.ones_like(mask, dtype=torch.float32, device=mask.device)
            depth_loss = L.direct.calculate_berhu_loss(equi_outputs, depth,                
               mask=mask, weights=attention_weights)      
            #gt_normal = depth2normal_gpu(depth)
            #pred_normal = depth2normal_gpu(equi_outputs)           
            #normal_loss = 1 - torch.mean(torch.sum((pred_normal * gt_normal * mask), dim=[1, 2, 3], keepdim=True) / mask.sum())
            #gt_grad = imgrad_yx(depth)
            #pred_grad = imgrad_yx(equi_outputs)
            #grad_loss = L.direct.calculate_l1_loss(pred_grad, gt_grad, mask)
            loss = depth_loss #+ normal_loss * 0.2 + grad_loss * 0.05

            rgb_img = rgb.detach().cpu().numpy()
            depth_prediction = equi_outputs.detach().cpu().numpy()
            equi_gt = depth.detach().cpu().numpy()
            depth_prediction[depth_prediction > 8] = 0
            if batch_idx % visualize_interval == 0:
                writer.add_image('RGB', vutils.make_grid(rgb[:2, [2,1,0], :, :].data, nrow=4, normalize=True), batch_idx)
                writer.add_image('depth gt', colorize(vutils.make_grid(depth[:2, ...].data, nrow=4, normalize=False)), batch_idx)
                writer.add_image('depth pred', colorize(vutils.make_grid(equi_outputs[:2, ...].data, nrow=4, normalize=False)), batch_idx)
                writer.add_image('error', colorize(vutils.make_grid(error[:2, ...].data, nrow=4, normalize=False)), batch_idx) 
                #writer.add_image('normal', vutils.make_grid(pred_normal[:2, ...].data, nrow=4, normalize=True), batch_idx)
                #writer.add_image('normal gt', vutils.make_grid(gt_normal[:2, ...].data, nrow=4, normalize=True), batch_idx)
                #writer.add_image('confidence mask', colorize(vutils.make_grid(weight[:8, ...].data, nrow=4, normalize=False)), batch_idx) 
                #writer.add_image('weight', colorize(vutils.make_grid(zero_weight[:4, ...].data, nrow=4, normalize=False)), batch_idx) 
                #writer.add_image('depth coarse', colorize(vutils.make_grid(coarse_outputs[:2, ...].data, nrow=4, normalize=False)), batch_idx)    
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            #scheduler.step()
            total_train_loss += loss.item()
            total_depth_loss += depth_loss.item()
            #total_normal_loss += normal_loss.item()*0.2
            #total_grad_loss += grad_loss.item()*0.05
            global_step += 1
            if batch_idx % visualize_interval == 0 and batch_idx > 0:
                print('[Epoch %d--Iter %d]depth loss %.4f' % 
                (epoch, batch_idx, total_depth_loss/(batch_idx+1)))


        print('lr for epoch ', epoch, ' ', optimizer.param_groups[0]['lr'])
        torch.save(network.state_dict(), os.path.join(args.save_path, args.save_checkpoint)+'/checkpoint_latest.pth')
        #-----------------------------------------------------------------------------
        scheduler.step()
        # Valid ----------------------------------------------------------------------------------------------------
        if epoch % 2 == 0:
            print('-------------Validate Epoch', epoch, '-----------')
            network.eval()
            for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
                bs, _, h, w = rgb.shape
                rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
        

                with torch.no_grad():
                    equi_outputs = network(rgb)
                    error = torch.abs(depth - equi_outputs) * mask
                    error[error < 0.1] = 0
                
                rgb_img = rgb.detach().cpu().numpy()
                depth_prediction = equi_outputs.detach().cpu().numpy()
                equi_gt = depth.detach().cpu().numpy()
                error_img = error.detach().cpu().numpy()
                depth_prediction[depth_prediction > 8] = 0
                
                # save raw 3D point cloud reconstruction as ply file
                coords = np.stack(np.meshgrid(range(w), range(h)), -1)
                coords = np.reshape(coords, [-1, 2])
                coords += 1
                uv = coords2uv(coords, w, h)          
                xyz = uv2xyz(uv) 
                xyz = torch.from_numpy(xyz).to(rgb.device)
                xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
                gtxyz = xyz * depth.reshape(bs, w*h, 1)    
                predxyz = xyz * equi_outputs.reshape(bs, w*h, 1)
                gtxyz = gtxyz.detach().cpu().numpy()
                predxyz = predxyz.detach().cpu().numpy()   
                #error = error.detach().cpu().numpy()
                if batch_idx % 20 == 0:
                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                    depth_pred_img = depth_prediction[0, 0, :, :]
                    depth_gt_img = equi_gt[0, 0, :, :]
                    error_img = error_img[0, 0, :, :] 
                    gtxyz_np = predxyz[0, ...]
                    predxyz_np = predxyz[0, ...]
                    cv2.imwrite('{}/test_equi_rgb_{}.png'.format(result_view_dir, batch_idx),
                                rgb_img*255)
                    plot.imsave('{}/test_equi_pred_{}.png'.format(result_view_dir, batch_idx),
                                depth_pred_img, cmap="jet")
                    plot.imsave('{}/test_equi_gt_{}.png'.format(result_view_dir, batch_idx),
                                depth_gt_img, cmap="jet")
                    plot.imsave('{}/test_error_{}.png'.format(result_view_dir, batch_idx),
                                error_img, cmap="jet")
                    rgb_img = np.reshape(rgb_img*255, (-1, 3)).astype(np.uint8)
                    write_ply('{}/test_gt_{}'.format(result_view_dir, batch_idx), [gtxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                    write_ply('{}/test_pred_{}'.format(result_view_dir, batch_idx), [predxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                #equi_mask *= mask            
                compute_eval_metrics(equi_outputs, depth, mask)

                global_val+=1
                #------------
            print('Epoch: {}\n'
            '  Avg. Abs. Rel. Error: {:.4f}\n'
            '  Avg. Sq. Rel. Error: {:.4f}\n'
            '  Avg. Lin. RMS Error: {:.4f}\n'
            '  Avg. Log RMS Error: {:.4f}\n'
            '  Inlier D1: {:.4f}\n'
            '  Inlier D2: {:.4f}\n'
            '  Inlier D3: {:.4f}\n\n'.format(
            epoch, 
            abs_rel_error_meter.avg,
            sq_rel_error_meter.avg,
            math.sqrt(lin_rms_sq_error_meter.avg),
            math.sqrt(log_rms_sq_error_meter.avg),
            d1_inlier_meter.avg,
            d2_inlier_meter.avg,
            d3_inlier_meter.avg))
            row = [epoch, '{:.4f}'.format(abs_rel_error_meter.avg.item()), 
                '{:.4f}'.format(sq_rel_error_meter.avg.item()), 
                '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter.avg).item()),
                '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter.avg).item()), 
                '{:.4f}'.format(d1_inlier_meter.avg.item()), 
                '{:.4f}'.format(d2_inlier_meter.avg.item()), 
                '{:.4f}'.format(d3_inlier_meter.avg.item()),
                '{:.8f}'.format(optimizer.param_groups[0]['lr'])]
            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)  
                csvwriter.writerow(row)
            writer.add_scalar('abs rel', abs_rel_error_meter.avg, epoch)
            writer.add_scalar('log rmse', math.sqrt(log_rms_sq_error_meter.avg), epoch)
            abs_rel_error_meter.reset()
            sq_rel_error_meter.reset()
            lin_rms_sq_error_meter.reset()
            log_rms_sq_error_meter.reset()
            d1_inlier_meter.reset()
            d2_inlier_meter.reset()
            d3_inlier_meter.reset()
            if abs_rel_error_meter.avg.item() < min_error:
                torch.save(network.state_dict(), os.path.join(args.save_path, args.save_checkpoint)+'/checkpoint_best.pth')
        
    # End Training
    print("Training Ended")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    writer.close()
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        