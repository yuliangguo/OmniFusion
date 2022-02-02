import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import time
from os import makedirs
from os.path import join, exists

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pers2equi(pers_img, fov, patch_size, erp_size):
    bs = pers_img.shape[0]
    grid_dir = './grid'
    if not exists(grid_dir):
        makedirs(grid_dir)
    grid_file = join(grid_dir, 'grid.pth')  
    if not exists(grid_file):  
        height, width = pair(patch_size)
        fov_h, fov_w = pair(fov)
        erp_h, erp_w = pair(erp_size)
        FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

        PI = math.pi
        PI_2 = math.pi * 0.5
        PI2 = math.pi * 2

        #num_rows = 6
        #num_cols = [3, 8, 12, 12, 8, 3]
        #phi_centers = [-75, -45, -15, 15, 45, 75]
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
        num_patch = all_combos.shape[0]

        center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
        center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
        center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

        cp = center_point * 2 - 1
        cp[:, 0] = cp[:, 0] * PI
        cp[:, 1] = cp[:, 1] * PI_2
        cp = cp.unsqueeze(1)

        lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
        lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
        lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 

        cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
        new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
        new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
        new_x = new_x / FOV[0] / PI   # -1 to 1
        new_y = new_y / FOV[1] / PI_2    

        new_x = (new_x + 1) * 0.5 * height
        new_y = (new_y + 1) * 0.5 * width
        new_x = new_x.reshape(num_patch, erp_h, erp_w) 
        new_y = new_y.reshape(num_patch, erp_h, erp_w) 

        new_x = new_x + shifts.reshape(-1, 1, 1)

        new_x *= erp_mask
        new_y *= erp_mask
        new_x = new_x.sum(0)
        new_y = new_y.sum(0)

        new_x = (new_x / pers_img.shape[-1] - 0.5) * 2
        new_y = (new_y / pers_img.shape[-2] - 0.5) * 2

        new_grid = torch.stack([new_x, new_y], -1)
        torch.save(new_grid, grid_file)
    else:
        new_grid = torch.load(grid_file)    
    new_grid = new_grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(pers_img.device)
    img_erp = F.grid_sample(pers_img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return img_erp

