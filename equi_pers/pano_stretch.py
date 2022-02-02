import torch
import torch.nn as nn
import torch.nn.functional as F 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import math
from scipy.ndimage import map_coordinates
import functools

def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv

@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v

def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5

def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y    

def pano_stretch(pano_img, kx, ky):
    img_w, img_h = pano_img.shape[-1], pano_img.shape[-2]
    sin_u, cos_u, tan_v = uv_tri(img_w, img_h)
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    x = u0 / np.pi
    y = v0 / (np.pi / 2)
    coord = torch.from_numpy(np.stack([x, y], axis=-1)).float()
    coord = coord.unsqueeze(0).to(pano_img.device)
    new_stretch = F.grid_sample(pano_img, coord, mode='bilinear', padding_mode='zeros', align_corners=True)
    return new_stretch   