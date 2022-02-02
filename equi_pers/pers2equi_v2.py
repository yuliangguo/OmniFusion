import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
#from equi2pers_torch_v2 import equi2pers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pers2equi(pers_img, fov, patch_size, erp_size, num_rows, num_cols):
    bs = pers_img.shape[0]
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    erp_h, erp_w = pair(erp_size)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2

    #num_rows = erp_h // height
    #num_cols = erp_w // width
    rows = np.linspace(-90.0, 90.0, num_rows + 1)
    rows = (rows[:-1] + rows[1:]) * 0.5
    cols = np.linspace(-180.0, 180.0, num_cols + 1)
    cols = (cols[:-1] + cols[1:]) * 0.5

    all_combos = []
    shifts = []
    u = (rows + 90) / 180 * erp_h
    v = (cols + 180) / 360 * erp_w
    erp_mask = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            all_combos.append([col, row])
            shifts.append([j*width, i*height])
            mask_tmp = np.zeros([erp_h, erp_w], dtype=int)
            up_u = int(u[i] + height//2)
            down_u = int(u[i] - height//2)
            up_v = int(v[j] + width//2)
            down_v = int(v[j] - width//2)
            mask_tmp[down_u:up_u, down_v:up_v] = 1 
            erp_mask.append(mask_tmp)

    all_combos = np.vstack(all_combos)
    center_point = torch.from_numpy(all_combos).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()

    center_point[:, 0] = (center_point[:, 0] + 180) / 360
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  

    cp = center_point * 2 - 1
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)  

    shifts = np.stack(shifts)
    shifts = torch.from_numpy(shifts).float()
    shifts = shifts.reshape(num_rows*num_cols, 1, 2)    

    lat_grid, lon_grid = torch.meshgrid(torch.linspace(-PI_2, PI_2, erp_h), torch.linspace(-PI, PI, erp_w))
    lon_grid = lon_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1)
    lat_grid = lat_grid.float().reshape(1, -1)#.repeat(num_rows*num_cols, 1) 

    cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(lon_grid - cp[..., 0])
    new_x = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0])) / cos_c
    new_y = (torch.cos(cp[..., 1])*torch.sin(lat_grid) - torch.sin(cp[...,1])*torch.cos(lat_grid)*torch.cos(lon_grid-cp[...,0])) / cos_c
    new_x = new_x / FOV[0] / PI   # -1 to 1
    new_y = new_y / FOV[1] / PI_2
    #mask = (new_x <= 1) & (new_x >= -1) & (new_y <= 1) & (new_y >= -1)
    #new_x = new_x * mask.float() 
    #new_y = new_y * mask.float() 

    new_x = (new_x + 1) * 0.5 * height
    new_y = (new_y + 1) * 0.5 * width

    new_x = new_x + shifts[..., 0]
    new_y = new_y + shifts[..., 1]

    new_x = new_x.reshape(num_rows*num_cols, erp_h, erp_w) * erp_mask
    new_y = new_y.reshape(num_rows*num_cols, erp_h, erp_w) * erp_mask

    new_x = new_x.sum(0)
    new_y = new_y.sum(0)

    new_x = (new_x / erp_w - 0.5) * 2
    new_y = (new_y / erp_h - 0.5) * 2

    new_grid = torch.stack([new_x, new_y], -1)
    new_grid = new_grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(pers_img.device)
    img_erp = F.grid_sample(pers_img, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return img_erp

if __name__ == '__main__':
    img = cv2.imread('pano5.png', cv2.IMREAD_COLOR)
    img_new = img.astype(np.float32) 
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    pers = equi2pers(img_new, fov=48, patch_size=(64, 64))
    img_erp = pers2equi(pers, fov=48, patch_size=(64, 64), erp_size=(256, 512))
    img_erp = img_erp[0].numpy()
    img_erp = img_erp.transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite('erp.png', img_erp)
    print(img_erp.shape)
