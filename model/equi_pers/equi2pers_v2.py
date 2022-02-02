import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def equi2pers(erp_img, fov, patch_size, num_rows, num_cols):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

    #num_rows = erp_h // height
    #num_cols = erp_w // width
    rows = np.linspace(-90.0, 90.0, num_rows + 1)
    rows = (rows[:-1] + rows[1:]) * 0.5
    cols = np.linspace(-180.0, 180.0, num_cols + 1)
    cols = (cols[:-1] + cols[1:]) * 0.5

    all_combos = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            all_combos.append([col, row])
    all_combos = np.vstack(all_combos)
    center_point = torch.from_numpy(all_combos).float()
    center_point[:, 0] = (center_point[:, 0] + 180) / 360
    center_point[:, 1] = (center_point[:, 1] + 90) / 180       

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

    lon_new = lon_new.view(num_rows, num_cols, height, width).permute(0, 2, 1, 3).contiguous().view(num_rows*height, num_cols*width)
    lat_new = lat_new.view(num_rows, num_cols, height, width).permute(0, 2, 1, 3).contiguous().view(num_rows*height, num_cols*width)    

    grid = torch.stack([lon_new, lat_new], -1)
    grid = torch.reshape(grid, [erp_h, erp_w, 2])
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return pers

if __name__ == '__main__':
    img = cv2.imread('pano5.png', cv2.IMREAD_COLOR)
    img_new = img.astype(np.float32) 
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    pers = equi2pers(img_new, fov=(52, 52), patch_size=(64, 64))
    pers = pers[0].numpy()
    pers = pers.transpose(1, 2, 0).astype(np.uint8)
    #cv2.imwrite('pers.png', pers)
    print(pers.shape)
