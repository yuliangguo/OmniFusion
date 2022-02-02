import os
import sys
import cv2
from math import pi
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

img = cv2.imread('pano4.png', cv2.IMREAD_COLOR)
#img = img.astype(np.float32) / 255
[erp_h, erp_w, _] = img.shape
bs = 2
img_new = img.astype(np.float32) / 255
img_new = np.transpose(img_new, [2, 0, 1])
img_new = torch.from_numpy(img_new)
img_new = img_new.unsqueeze(0).repeat(bs, 1, 1, 1)

height, width = 96, 96
FOV = [90, 90]
FOV = [FOV[0]/360.0, FOV[1]/180.0]
FOV = torch.tensor(FOV, dtype=torch.float32)
PI = math.pi
PI_2 = math.pi * 0.5
PI2 = math.pi * 2
yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

num_rows = 4
num_cols = [3, 6, 6, 3]
phi_centers = [-67.5, -22.5, 22.5, 67.5]
phi_interval = 180 // num_rows
all_combos = []
erp_mask = []
for i, n_cols in enumerate(num_cols):
    for j in np.arange(n_cols):
        theta_interval = 360 / n_cols
        theta_center = j * theta_interval + theta_interval / 2
        center = [theta_center, phi_centers[i]]
        all_combos.append(center)
        up = phi_centers[i] + phi_interval / 2
        down = phi_centers[i] - phi_interval / 2
        left = theta_center - theta_interval / 2
        right = theta_center + theta_interval / 2
        up = int((up + 90) / 180 * erp_h)
        down = int((down + 90) / 180 * erp_h)
        left = int(left / 360 * erp_w)
        right = int(right / 360 * erp_w)
        mask = np.zeros((erp_h, erp_w), dtype=int)
        mask[down:up, left:right] = 1
        erp_mask.append(mask)
all_combos = np.vstack(all_combos) 
shifts = np.arange(all_combos.shape[0]) * width
shifts = torch.from_numpy(shifts).float()
erp_mask = np.stack(erp_mask)
erp_mask = torch.from_numpy(erp_mask).float()
n_patch = all_combos.shape[0]

center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1
cp = center_point * 2 - 1
cp[:, 0] = cp[:, 0] * PI
cp[:, 1] = cp[:, 1] * PI_2
cp = cp.unsqueeze(1)
convertedCoord = screen_points * 2 - 1
convertedCoord[:, 0] = convertedCoord[:, 0] * PI
convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

x = convertedCoord[:, :, 0]
y = convertedCoord[:, :, 1]

rou = torch.sqrt(x ** 2 + y ** 2)
c = torch.atan(rou)
sin_c = torch.sin(c)
cos_c = torch.cos(c)
lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
lat_new = lat / PI_2 
lon_new = lon / PI 
lon_new[lon_new > 1] -= 2
lon_new[lon_new<-1] += 2 

lon_new = lon_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
lat_new = lat_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
grid = torch.stack([lon_new, lat_new], -1)

grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1)
persp = F.grid_sample(img_new, grid, mode='bilinear', padding_mode='border', align_corners=True)

persp_reshape = F.unfold(persp, kernel_size=(height, width), stride=(height, width))
persp_reshape = persp_reshape.reshape(bs, 3, height, width, n_patch)

lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 
cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
new_x = new_x / FOV[0] / PI   # -1 to 1
new_y = new_y / FOV[1] / PI_2
cos_c_mask = cos_c.reshape(n_patch, erp_h, erp_w)
cos_c_mask = torch.where(cos_c_mask > 0, 1, 0)


w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32)


new_x_patch = (new_x + 1) * 0.5 * height
new_y_patch = (new_y + 1) * 0.5 * width 
new_x_patch = new_x_patch.reshape(n_patch, erp_h, erp_w)
new_y_patch = new_y_patch.reshape(n_patch, erp_h, erp_w)
# valid mask
mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
mask *= cos_c_mask

x0 = torch.floor(new_x_patch).type(torch.int64)
x1 = x0 + 1
y0 = torch.floor(new_y_patch).type(torch.int64)
y1 = y0 + 1
    
x0 = torch.clamp(x0, 0, width-1)
x1 = torch.clamp(x1, 0, width-1)
y0 = torch.clamp(y0, 0, height-1)
y1 = torch.clamp(y1, 0, height-1)

wa = (x1.type(torch.float32)-new_x_patch) * (y1.type(torch.float32)-new_y_patch)
wb = (x1.type(torch.float32)-new_x_patch) * (new_y_patch-y0.type(torch.float32))
wc = (new_x_patch-x0.type(torch.float32)) * (y1.type(torch.float32)-new_y_patch)
wd = (new_x_patch-x0.type(torch.float32)) * (new_y_patch-y0.type(torch.float32))

wa = wa * mask.expand_as(wa)
wb = wb * mask.expand_as(wb)
wc = wc * mask.expand_as(wc)
wd = wd * mask.expand_as(wd)

w_list[..., 0] = wa
w_list[..., 1] = wb
w_list[..., 2] = wc
w_list[..., 3] = wd

# x shape [18, 256, 512]
z = torch.arange(n_patch)
z = z.reshape(n_patch, 1, 1)

Ia = persp_reshape[:, :, y0, x0, z]
Ib = persp_reshape[:, :, y1, x0, z]
Ic = persp_reshape[:, :, y0, x1, z]
Id = persp_reshape[:, :, y1, x1, z]

output_a = Ia * mask.expand_as(Ia)
output_b = Ib * mask.expand_as(Ib)
output_c = Ic * mask.expand_as(Ic)
output_d = Id * mask.expand_as(Id)

output_a = output_a.permute(0, 1, 3, 4, 2)
output_b = output_b.permute(0, 1, 3, 4, 2)
output_c = output_c.permute(0, 1, 3, 4, 2)
output_d = output_d.permute(0, 1, 3, 4, 2)


"""
w_list = torch.zeros((n_patch, erp_h, erp_w, 4), dtype=torch.float32)
output_a = torch.zeros((bs, 3, erp_h, erp_w, n_patch), dtype=torch.float32)
output_b = torch.zeros_like(output_a, dtype=torch.float32)
output_c = torch.zeros_like(output_a, dtype=torch.float32)
output_d = torch.zeros_like(output_a, dtype=torch.float32)
for n_p in range(n_patch):
    new_x_patch = (new_x[n_p, ...] + 1) * 0.5 * height
    new_y_patch = (new_y[n_p, ...] + 1) * 0.5 * width
    new_x_patch = new_x_patch.reshape(erp_h, erp_w) 
    new_y_patch = new_y_patch.reshape(erp_h, erp_w) 
    mask = torch.where((new_x_patch < width) & (new_x_patch > 0) & (new_y_patch < height) & (new_y_patch > 0), 1, 0)
    mask *= cos_c_mask[n_p, ...]
    
    one_patch = persp_reshape[..., n_p]
    
    x0 = torch.floor(new_x_patch).type(torch.int64)
    x1 = x0 + 1
    y0 = torch.floor(new_y_patch).type(torch.int64)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    y0 = torch.clamp(y0, 0, height-1)
    y1 = torch.clamp(y1, 0, height-1)

    wa = (x1.type(torch.float32)-new_x_patch) * (y1.type(torch.float32)-new_y_patch)
    wb = (x1.type(torch.float32)-new_x_patch) * (new_y_patch-y0.type(torch.float32))
    wc = (new_x_patch-x0.type(torch.float32)) * (y1.type(torch.float32)-new_y_patch)
    wd = (new_x_patch-x0.type(torch.float32)) * (new_y_patch-y0.type(torch.float32))
    
    wa = wa * mask.expand_as(wa)
    wb = wb * mask.expand_as(wb)
    wc = wc * mask.expand_as(wc)
    wd = wd * mask.expand_as(wd)
    
    w_list[n_p, ..., 0] = wa
    w_list[n_p, ..., 1] = wb
    w_list[n_p, ..., 2] = wc
    w_list[n_p, ..., 3] = wd

    Ia = one_patch[:, :, y0, x0]
    Ib = one_patch[:, :, y1, x0]
    Ic = one_patch[:, :, y0, x1]
    Id = one_patch[:, :, y1, x1]
    
    Ia = Ia * mask.expand_as(Ia)
    Ib = Ib * mask.expand_as(Ib)
    Ic = Ic * mask.expand_as(Ic)
    Id = Id * mask.expand_as(Id)
    
    output_a[..., n_p] = Ia
    output_b[..., n_p] = Ib
    output_c[..., n_p] = Ic
    output_d[..., n_p] = Id
"""

w_list = w_list.permute(1, 2, 0, 3)
w_list = w_list.flatten(2)
w_list *= torch.gt(w_list, 1e-5).type(torch.float32)
w_list = F.normalize(w_list, p=1, dim=-1).reshape(erp_h, erp_w, n_patch, 4)
#w_list = w_list.permute(2, 0, 1, 3)

output = output_a * w_list[..., 0].expand_as(output_a) + output_b * w_list[..., 1].expand_as(output_b) + \
    output_c * w_list[..., 2].expand_as(output_c) + output_d * w_list[..., 3].expand_as(output_d)
output = output.sum(-1)    

img_erp_int = output[1, ...].permute(1, 2, 0).numpy()
img_erp_int = img_erp_int * 255
img_erp_int = img_erp_int.astype(np.uint8)
cv2.imwrite('erp.png', img_erp_int)
    
    
    
    
    