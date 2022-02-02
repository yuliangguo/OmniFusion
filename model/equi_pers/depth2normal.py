import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
from util import uv2coords, uv2xyz, coords2uv

def depth2normal(depth):
    device = depth.device
    bs, _, h, w = depth.shape
    depth = depth.reshape(bs, h, w)
    coords = np.stack(np.meshgrid(range(w), range(h)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = coords2uv(coords, w, h)          
    xyz = uv2xyz(uv) 
    xyz = torch.from_numpy(xyz).to(device)
    xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
    
    depth_reshape = depth.reshape(bs, h*w, 1)
    newxyz = xyz * depth_reshape
    newxyz_reshape = newxyz.reshape(bs, h, w, 3).permute(0, 3, 1, 2)  #bs x 3 x h x w
    kernel_size = 5
    point_matrix = F.unfold(newxyz_reshape, kernel_size=kernel_size, stride=1, padding=kernel_size-1, dilation=2)
    
    # An = b 
    matrix_a = point_matrix.view(bs, 3, kernel_size*kernel_size, h, w)  # (B, 3, 25, H, W)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, H, W, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4) # (b, h, w, 3, 25)
    matrix_b = torch.ones([bs, h, w, kernel_size*kernel_size, 1], device=device)
    
    #normal = torch.linalg.lstsq(matrix_a, matrix_b)
    #normal = normal.solution.squeeze(-1)
    #norm_normalize = F.normalize(normal, p=2, dim=-1)
    #norm_normalize = norm_normalize.permute(0, 3, 1, 2)
    
    
    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)  # (b, h, w, 3, 3)
    matrix_deter = torch.det(point_multi)
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32, device=device)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(bs, h, w, 1, 1)
    # inversible matrix

    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.linalg.inv(inversible_matrix)
    
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
  
    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    norm_normalize = norm_normalize.squeeze(-1)
    norm_normalize = norm_normalize.permute(0, 3, 1, 2)
    
    return norm_normalize
    