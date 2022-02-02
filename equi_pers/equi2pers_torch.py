import os
import sys
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 


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


def equi2pers(equi_img, hFOV, wFOV, theta, phi, output_h, output_w):
    """[torch version of equirectangular to perspective]

    Args:
        pers_img ([float image]): [input equi image, batch x 3 x h x w]
        hFOV ([int]): [horizontal FOV]
        wFOV ([int]): [vertical FOV]
        theta ([int]): [theta angle, the center of image theta = 0]
        phi ([int]): [phi angle, the center of image phi = 0]
        output_h ([int]): [output pers height]
        output_w ([int]): [output pers width]
    """    
    [bs, _, height, width] = equi_img.shape
    num_patch = len(theta)
    w_len = math.tan(math.radians(wFOV / 2.0))
    h_len = math.tan(math.radians(hFOV / 2.0))
    equ_h = height
    equ_w = width
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0
    x_map = torch.ones([output_h, output_w], dtype=torch.float32)
    y_map = torch.linspace(-w_len, w_len, output_w).unsqueeze(0).repeat(output_h, 1)
    z_map = -torch.linspace(-h_len, h_len, output_h).unsqueeze(0).repeat(output_w, 1).transpose(1, 0)

    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)  #bs x h x w
    xyz = torch.stack((x_map,y_map,z_map),dim=-1)/ D.unsqueeze(-1)  #bs x h x w x 3

    y_axis = torch.as_tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=equi_img.device)
    z_axis = torch.as_tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=equi_img.device)

    R1 = rotation_matrix(torch.deg2rad(theta), z_axis) 
    R2 = torch.matmul(R1, y_axis.view(1, 3, 1)).squeeze(-1)
    R2 = rotation_matrix(torch.deg2rad(-phi), R2)  # bs x 3 x 3*3,

    xyz = xyz.view(output_h * output_w, 3).transpose(1, 0).to(equi_img.device)   # bs x 3 x h*w
    xyz = torch.matmul(R1, xyz)                    # bs x 3 x h*w
    xyz = torch.matmul(R2, xyz).transpose(2, 1)    # bs x h*w x 3
    lat = torch.asin(xyz[..., 2])
    lon = torch.atan2(xyz[..., 1] , xyz[..., 0])

    lon = lon / math.pi * 180   
    lat = -lat / math.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90  * equ_cy + equ_cy

    lon = (lon / width - 0.5)*2
    lat = (lat / height - 0.5)*2
    
    lon = lon.view(1, num_patch, output_h, output_w).permute(0, 2, 1, 3).contiguous().view(output_h, num_patch*output_w)
    lat = lat.view(1, num_patch, output_h, output_w).permute(0, 2, 1, 3).contiguous().view(output_h, num_patch*output_w)

    grid = torch.stack([lon, lat], -1)   #bs x h x w x 2
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1)
    grid = grid.to(equi_img.device)
    #grid = grid.cuda()
    persp = F.grid_sample(equi_img, grid, mode='bilinear', padding_mode='zeros',align_corners=True)   # bs x 3 x h x w

    return persp

if __name__ == "__main__":
    img = cv2.imread('test.png', -1).astype(np.float32)
    img = img[...,:3] / 255. 
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    print(img.shape)
    theta = [0, 20, 30, -20]
    phi = [0, 10, 20, 10]
    bs = 2
    theta = torch.tensor(theta, dtype=torch.float32)
    phi = torch.tensor(phi, dtype=torch.float32)
    img = img.repeat(bs, 1, 1, 1)
    persp = equi2pers(img, 80, 80, theta, phi, 320, 320)
    print(img.shape, persp.shape)
    persp_img = persp[0, :, :, :].permute(1,2,0)
    persp_img = persp_img.numpy()
    persp_img = persp_img * 255
    plt.imshow(persp_img[:,:,[2,1,0]].astype(np.uint8))
    plt.show()