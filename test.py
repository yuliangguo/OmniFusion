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
from dataset_loader_stanford import Dataset
import cv2
import supervision as L
import spherical as S360
from util import load_partial_model
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
import scipy.io
#from model_spherical import Network
from network_test import spherical_fusion
#from model.spherical_fusion import *
from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils 
import equi_pers.equi2pers_v3
from thop import profile

parser = argparse.ArgumentParser(description='360Transformer')
parser.add_argument('--input_dir', default='/media/rtx2/DATA/stanford2d3d',
#parser.add_argument('--input_dir', default='/home/rtx2/NeurIPS/spherical_mvs/data/omnidepth',
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
parser.add_argument('--save_path', default='./stanford/512x1024/resnet34/visualize_baseline',
                    help='save checkpoint path')                    
parser.add_argument('--tensorboard_path', default='logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


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
ckpt = torch.load(args.save_path + '/checkpoints/checkpoint_latest.tar')

#network.load_state_dict(ckpt)
network.cuda()

#----------------------------------------------------------

print('## Batch size: {}'.format(batch_size))
print('## learning rate: {}'.format(init_lr))  
print('## patch size:', patch_size) 
print('## fov:', args.fov)
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in network.parameters() if p.requires_grad is True])))
#--------------------------------------------------



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
 
 
abs_rel_error_meter = AverageMeter()
sq_rel_error_meter = AverageMeter()
lin_rms_sq_error_meter = AverageMeter()
log_rms_sq_error_meter = AverageMeter()
d1_inlier_meter = AverageMeter()
d2_inlier_meter = AverageMeter()
d3_inlier_meter = AverageMeter()

network.eval()
input = torch.randn(1, 3, 512, 1024).cuda()
with torch.no_grad():
    for i in tqdm(range(55)):
        if i == 5:
            start_time = time.time()
        patch_depth = network(input, fov, patch_size, nrows, 2)

end_time = time.time()
total_forward = end_time - start_time
print('Total forward time is %4.2f seconds' % total_forward)
actual_num_runs = 50
latency = total_forward / actual_num_runs
fps = 1 * actual_num_runs / total_forward
print("Latency: ", latency, "fps", fps)
exit()      
# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0
    result_view_dir = 'feature_output'
    network.eval()
    for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
        bs, _, h, w = rgb.shape
        rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
        
        with torch.no_grad():
            equi_outputs_list, patch, layer1, patch_depth = network(rgb, fov, patch_size, nrows, 2)
            equi_outputs = equi_outputs_list[-1]
            patch = F.fold(patch.reshape(bs, -1, 18), output_size=(256, 256*18), kernel_size=256, stride=256)
            patch_depth = F.fold(patch_depth.reshape(bs, -1, 18), output_size=(256, 256*18), kernel_size=256, stride=256)
            #layer1 = F.fold(layer1.reshape(bs, -1, 18), output_size=(64, 64*18), kernel_size=64, stride=64)

            compute_eval_metrics(equi_outputs, depth, mask)
            rgb_img = rgb.detach().cpu().numpy()
            depth_prediction = equi_outputs.detach().cpu().numpy()
            patch = patch.detach().cpu().numpy()
            patch_depth = patch_depth.detach().cpu().numpy()
            layer1 = layer1.detach().cpu().numpy()
            if batch_idx % 10 == 0:
                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                    depth_pred_img = depth_prediction[0, 0, :, :]
                    layer1_img = layer1[0, :, :, :].mean(0)
                    
                    patch_img = patch[0, :, :, :].transpose(1, 2, 0)
                    patch_depth_img = patch_depth[0, 0, ...]
                    cv2.imwrite('{}/test_equi_rgb_{}.png'.format(result_view_dir, batch_idx),
                                rgb_img*255)
                    cv2.imwrite('{}/test_patch_rgb_{}.png'.format(result_view_dir, batch_idx),
                                patch_img*255)
                    plot.imsave('{}/test_equi_pred_{}.png'.format(result_view_dir, batch_idx),
                                depth_pred_img, cmap="jet")
                    plot.imsave('{}/test_patch_pred_{}.png'.format(result_view_dir, batch_idx),
                                patch_depth_img, cmap="jet")
                    plot.imsave('{}/test_feature_{}.png'.format(result_view_dir, batch_idx),
                                layer1_img, cmap="plasma")
    print(
    '  Avg. Abs. Rel. Error: {:.4f}\n'
    '  Avg. Sq. Rel. Error: {:.4f}\n'
    '  Avg. Lin. RMS Error: {:.4f}\n'
    '  Avg. Log RMS Error: {:.4f}\n'
    '  Inlier D1: {:.4f}\n'
    '  Inlier D2: {:.4f}\n'
    '  Inlier D3: {:.4f}\n\n'.format(
    abs_rel_error_meter.avg,
    sq_rel_error_meter.avg,
    math.sqrt(lin_rms_sq_error_meter.avg),
    math.sqrt(log_rms_sq_error_meter.avg),
    d1_inlier_meter.avg,
    d2_inlier_meter.avg,
    d3_inlier_meter.avg))

#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        