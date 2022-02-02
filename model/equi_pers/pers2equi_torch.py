import os
import sys
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .equi2pers_torch import equi2pers


def rotation_matrix(theta, axis):
    n = theta.shape[0]
    #axis = torch.as_tensor(axis, device='cpu', dtype=input.dtype)
    axis = F.normalize(axis, dim=-1)
    a = torch.cos(theta / 2.0)
    axis = axis.view(-1, 3)
    tmp = -axis * torch.sin(theta.view(n, 1) / 2.0)
    b, c, d = tmp[:, 0], tmp[:, 1], tmp[:, 2]
    #b, c, d = -axis * torch.sin(theta.view(n, 1) / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = torch.zeros([n, 3, 3], dtype=theta.dtype, device=theta.device)
    rotation_matrix[:, 0, 0] = aa + bb - cc - dd
    rotation_matrix[:, 0, 1] = 2 * (bc + ad)
    rotation_matrix[:, 0, 2] = 2 * (bd - ac)
    rotation_matrix[:, 1, 0] = 2 * (bc - ad)
    rotation_matrix[:, 1, 1] = aa + cc - bb - dd
    rotation_matrix[:, 1, 2] = 2 * (cd + ab)
    rotation_matrix[:, 2, 0] = 2 * (bd + ac)
    rotation_matrix[:, 2, 1] = 2 * (cd - ab)
    rotation_matrix[:, 2, 2] = aa + dd - bb - cc

    return rotation_matrix

def pers2equi(pers_img, hFOV, wFOV, theta, phi, output_h, output_w):
    [bs, _, h, w] = pers_img.shape
    w_len = math.tan(math.radians(wFOV / 2.0))
    h_len = math.tan(math.radians(hFOV / 2.0))

    y, x= torch.meshgrid(torch.linspace(90,-90, output_h), torch.linspace(-180, 180, output_w))
    x_map = torch.cos(torch.deg2rad(x)) * torch.cos(torch.deg2rad(y))
    y_map = torch.sin(torch.deg2rad(x)) * torch.cos(torch.deg2rad(y))
    z_map = torch.sin(torch.deg2rad(y))
    xyz = torch.stack((x_map,y_map,z_map),axis=2)
    xyz = xyz.unsqueeze(0).repeat(bs, 1, 1, 1).to(pers_img.device)
    y_axis = torch.as_tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=pers_img.device)
    z_axis = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=pers_img.device)

    R1 = rotation_matrix(torch.deg2rad(theta), z_axis) 
    R2 = torch.matmul(R1, y_axis.view(1, 3, 1)).squeeze(-1)
    R2 = rotation_matrix(torch.deg2rad(-phi), R2)  # bs x 3 x 3*3,
    R1 = torch.inverse(R1)
    R2 = torch.inverse(R2)

    xyz = xyz.view(bs, output_h * output_w, 3).transpose(2, 1)   # bs x 3 x h*w
    xyz = torch.matmul(R2, xyz)                    # bs x 3 x h*w
    xyz = torch.matmul(R1, xyz).transpose(2, 1)    # bs x h*w x 3
    
    xyz = xyz.view(bs, output_h, output_w, 3)
    inverse_mask = torch.where(xyz[:,:,:,0]>0,1,0)
    xyz[:,:,:] = xyz[:,:,:]/(xyz[:,:,:,0].unsqueeze(-1).repeat(1,1,1,3))

    lon_map = torch.where((-w_len<xyz[:,:,:,1])&(xyz[:,:,:,1]<w_len)&(-h_len<xyz[:,:,:,2])
                    &(xyz[:,:,:,2]<h_len),(xyz[:,:,:,1]+w_len)/2/w_len*float(w),torch.tensor(0, dtype=torch.float32, device=pers_img.device))
    lat_map = torch.where((-w_len<xyz[:,:,:,1])&(xyz[:,:,:,1]<w_len)&(-h_len<xyz[:,:,:,2])
                    &(xyz[:,:,:,2]<h_len),(-xyz[:,:,:,2]+h_len)/2/h_len*float(h),torch.tensor(0, dtype=torch.float32, device=pers_img.device))
    mask = torch.where((-w_len<xyz[:,:,:,1])&(xyz[:,:,:,1]<w_len)&(-h_len<xyz[:,:,:,2])
                    &(xyz[:,:,:,2]<h_len),1,0)

    lon_map = (lon_map / w - 0.5)*2
    lat_map = (lat_map / h - 0.5)*2 

    grid = torch.stack([lon_map, lat_map], -1)   
    device = pers_img.device
    grid = grid.to(device)
    sample_erp = F.grid_sample(pers_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)    
    mask *= inverse_mask
    mask = mask.unsqueeze(1).to(device)
    sample_erp *= mask
    
    return sample_erp, mask


if __name__ == "__main__":
    img = cv2.imread('test.png', -1).astype(np.float32)
    img = img[...,:3] / 255. 
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    print(img.shape)
    theta = [0, 20, 30, 0]  # -180 to 180
    phi = [0, 10, 20, 80]   # -90 to 90
    bs = len(phi)
    theta = torch.tensor(theta, dtype=torch.float32)
    phi = torch.tensor(phi, dtype=torch.float32)
    img = img.repeat(bs, 1, 1, 1)
    persp = equi2pers(img, 80, 80, theta, phi, 320, 320)
    equi, _ = pers2equi(persp, 80, 80, theta, phi, 512, 1024)
    equi_img = equi[3, :, :, :].permute(1,2,0)
    equi_img = equi_img.numpy()
    equi_img = equi_img * 255
    
    plt.imshow(equi_img[:,:,[2,1,0]].astype(np.uint8))
    plt.show()          