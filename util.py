import sys
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn import preprocessing
#import OpenEXR, Imath, array
import torch.nn.functional as F
import os
import os.path as osp
import shutil
import spherical as S360
import supervision as L
from equi_pers.equi2pers_v3 import equi2pers
from equi_pers.pers2equi_v3 import pers2equi
import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap='jet'):
    value = value.detach().cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def transform_equi(equi, THETA, PHI, output_h, output_w, select_pers, h_fov,
                   v_fov):
    [bs, _, h, w] = equi.shape
    #num_theta, num_phi = theta.shape[0], phi.shape[0]
    #THETA = theta.unsqueeze(0).repeat(bs, 1).flatten()
    #PHI = phi.unsqueeze(0).repeat(bs, 1).flatten()
    equi_repeat = equi.unsqueeze(1).repeat(1, select_pers, 1, 1, 1)
    equi_repeat = equi_repeat.view(bs * select_pers, -1, h, w)
    pers = equi2pers(equi_repeat, h_fov, v_fov, THETA, PHI, output_h, output_w)
    return pers


def transform_pers(pers, THETA, PHI, output_h_pano, output_w_pano, h_fov, v_fov):
   # bs = batch_size
    h, w = pers.shape[2], pers.shape[3]
    #num_theta, num_phi = theta.shape[0], phi.shape[0]
    #pers_repeat = pers.unsqueeze(0).repeat(bs,1,1,1,1)
    #pers_repeat = pers_repeat.view(bs*select_pers, -1, h, w)
    equi, mask = pers2equi(pers, h_fov, v_fov, THETA, PHI, output_h_pano,
                           output_w_pano)
    return equi, mask.unsqueeze(1)

def gaussian_kernel_1d(size, sigma):

    ax = np.linspace(-(size-1)/2, (size-1)/2, size)
    #xx, yy, zz = np.meshgrid(ax, ax, ax)  
    kernel = np.exp(-0.5 * (np.square(ax)) / np.square(sigma))
    
    return kernel

def gaussian_kernel_3d(size, sigma):

    ax = np.linspace(-(size-1)/2, (size-1)/2, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)  
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.square(sigma))
    
    return kernel

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def mkdirs(path):
    '''Convenience function to make all intermediate folders in creating a directory'''
    try:
        os.makedirs(path)
    except:
        pass


def xavier_init(m):
    '''Provides Xavier initialization for the network weights and 
    normally distributes batch norm params'''
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('ConvTranspose2d') != -1):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)

def save_checkpoint(state, is_best, filename):
    '''Saves a training checkpoints'''
    torch.save(state, filename)
    if is_best:
        basename = osp.basename(filename) # File basename
        idx = filename.find(basename) # Index where path ends and basename begins
        # Copy the file to a different filename in the same directory
        shutil.copyfile(filename, osp.join(filename[:idx], 'model_best.pth'))


def load_partial_model(model, loaded_state_dict):
    '''Loaded a save model, even if the model is not a perfect match. This will run even if there is are layers from the current network missing in the saved model. 
    However, layers without a perfect match will be ignored.'''
    model_dict = model.state_dict()
    pretrained_dict = {k : v for k,v in loaded_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Freeze Unfreeze Function 
# freeze_layer ----------------------
def freeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = False
# Unfreeze_layer --------------------
def unfreeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = True
 

def load_optimizer(optimizer, loaded_optimizer_dict, device):
    '''Loads the saved state of the optimizer and puts it back on the GPU if necessary.  Similar to loading the partial model, this will load only the optimization parameters that match the current parameterization.'''
    optimizer_dict = optimizer.state_dict()
    pretrained_dict = {k : v for k,v in loaded_optimizer_dict.items() 
        if k in optimizer_dict and k != 'param_groups'}
    optimizer_dict.update(pretrained_dict)
    optimizer.load_state_dict(optimizer_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)



def set_caffe_param_mult(m, base_lr, base_weight_decay):
    '''Function that allows us to assign a LR multiplier of 2 and a decay multiplier of 0 to the bias weights (which is common in Caffe)'''
    param_list = []
    for name, params in m.named_parameters():
        if name.find('bias') != -1:
            param_list.append({'params' : params, 'lr' : 2 * base_lr, 'weight_decay' : 0.0})
        else:
            param_list.append({'params' : params, 'lr' : base_lr, 'weight_decay' : base_weight_decay})
    return param_list



def coords2uv(coords, w, h):
    #output uv size w*h*2
    uv = np.zeros_like(coords, dtype = np.float32)
    middleX = w/2 + 0.5
    middleY = h/2 + 0.5
    uv[..., 0] = (coords[...,0] - middleX) / w * 2 * np.pi
    uv[..., 1] = -(coords[...,1] - middleY) / h * np.pi
    return uv


def uv2xyz(uv):
    xyz = np.zeros((uv.shape[0], 3), dtype = np.float32)
    xyz[:, 0] = np.multiply(np.cos(uv[:, 1]), np.sin(uv[:, 0]))
    xyz[:, 1] = np.multiply(np.cos(uv[:, 1]), np.cos(uv[:, 0]))
    xyz[:, 2] = np.sin(uv[:, 1])
    return xyz


def xyz2uv(xyz):
    normXY = torch.sqrt(xyz[:, :, 0]*xyz[:, :, 0] + xyz[:, :, 1]*xyz[:, :, 1])
    normXY[normXY < 1e-6] = 1e-6
    normXYZ = torch.sqrt(xyz[:, :, 0]*xyz[:, :, 0] + xyz[:, :, 1]*xyz[:, :, 1] + xyz[:, :, 2]*xyz[:, :, 2])
    
    v = torch.asin(xyz[:,:,2]/normXYZ)
    u = torch.asin(xyz[:,:,0]/normXY)
    valid = (xyz[:, :, 1] < 0) * ( u >= 0)
    u[valid] = math.pi - u[valid]
    valid = (xyz[:, :, 1] < 0) * ( u <= 0)
    u[valid] = -math.pi - u[valid]
    uv = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], -1)
    uv[uv!=uv] = 0  #remove nan

    return uv

def uv2coords(uv, w, h):
    coords = torch.zeros_like(uv, dtype = torch.float32, device = 'cuda:0')
    coords[...,0] = (uv[...,0] + math.pi)/2/math.pi * w + 0.5
    coords[...,1] = (math.pi/2 - uv[...,1])/math.pi * h + 0.5
    coords[...,0] = torch.min(coords[...,0], torch.cuda.FloatTensor([w]))
    coords[...,1] = torch.min(coords[...,1], torch.cuda.FloatTensor([h]))
    return coords

def chamfer_distance_with_batch(p1, p2, debug=False):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    if debug:
        print(p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))
        print(p1[0][0])

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 2)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0][0])

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0][0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=3)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=2)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist    

def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (H, W)
    coordinates: (2, ...)
    '''
    h = input.shape[0]
    w = input.shape[1]

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    f00 = input[co_floor[0], co_floor[1]]
    f10 = input[co_floor[0], co_ceil[1]]
    f01 = input[co_ceil[0], co_floor[1]]
    f11 = input[co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    output = fx1 + d2 * (fx2 - fx1) 
    return output  

def depth2normal(depth, h=256, w=512):
    #depth image size 256x512
    #return normal image 256x512x3
    #return curvature image 256x512
    coords = np.stack(np.meshgrid(range(512), range(256)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = coords2uv(coords, 512, 256)      
    xyz = uv2xyz(uv)                 #3d coordinates on a sphere
    depth = np.reshape(depth, [512*256, 1])
    depth = np.tile(depth, [1, 3])
    newxyz = np.multiply(xyz, depth)
    
    reshape_xyz = np.reshape(newxyz, [256, 512, 3])
    reshape_xyz = np.pad(reshape_xyz, ((1,1),(1,1),(0,0)), 'edge')
 
    vec0 = reshape_xyz[:h, 1:-1, :] - reshape_xyz[2:, 1:-1, :]
    vec2 = reshape_xyz[1:-1, :w, :] - reshape_xyz[1:-1, 2:, :]
    vec4 = reshape_xyz[2:, 1:-1, :] - reshape_xyz[:h, 1:-1, :]
    vec6 = reshape_xyz[1:-1, 2:, :] - reshape_xyz[1:-1, :w, :]
    
    normal = preprocessing.normalize(np.cross(np.reshape(vec2, [w*h, 3]), np.reshape(vec0, [w*h, 3])), norm = 'l2')
    normal += preprocessing.normalize(np.cross(np.reshape(vec4, [w*h, 3]), np.reshape(vec2, [w*h, 3])), norm = 'l2')
    normal += preprocessing.normalize(np.cross(np.reshape(vec6, [w*h, 3]), np.reshape(vec4, [w*h, 3])), norm = 'l2')
    normal += preprocessing.normalize(np.cross(np.reshape(vec0, [w*h, 3]), np.reshape(vec6, [w*h, 3])), norm = 'l2')
    
    normal = preprocessing.normalize(normal, norm='l2')

    normal = np.reshape(normal, [256, 512, 3])
    reshape_normal = np.pad(normal, ((1,1),(1,1),(0,0)), 'edge')

    n1 = reshape_normal[:h, 1:-1, :]
    n2 = reshape_normal[2:, 1:-1, :]
    n3 = reshape_normal[1:-1, :w, :]
    n4 = reshape_normal[1:-1, 2:, :]
    cur = (1 - np.einsum('ij,ij->i', np.reshape(n1, [w*h, 3]), np.reshape(n2, [w*h, 3])))/2
 
    cur += (1 - np.einsum('ij,ij->i', np.reshape(n3, [w*h, 3]), np.reshape(n4, [w*h, 3])))/2

    cur = cur/2
    cur = np.reshape(cur, [256, 512])
    cur[cur<1e-6] = 0
    normal = (normal+1)/2
    return normal, cur

def depth2normal_gpu(depth):
    """
    input: depth image, size Bx1xHxW
    output: derived boundary
    """
    batch, c, h, w = depth.shape
    depth = depth.view(batch, -1, 1)
    coords = np.stack(np.meshgrid(range(w), range(h)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = coords2uv(coords, w, h)          
    xyz = uv2xyz(uv)                 #3d coordinates on a sphere
    xyz = torch.from_numpy(xyz).cuda()
    xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)
    newxyz = xyz * depth

    vertices = newxyz.view(batch, h, w, 3).permute(0, 3, 1, 2)

    vec0_pad = F.pad(vertices[:, :, :, :-1] - vertices[:, :, :, 1:], pad=[0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
    vec2_pad = F.pad(vertices[:, :, :-1, :] - vertices[:, :, 1:, :], pad=[0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
    vec4_pad = F.pad(vertices[:, :, :, 1:] - vertices[:, :, :, :-1], pad=[1, 0, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
    vec6_pad = F.pad(vertices[:, :, 1:, :] - vertices[:, :, :-1, :], pad=[0, 0, 1, 0, 0, 0, 0, 0], mode='constant', value=0)

    # 4 connect
    cross20 = torch.cross(vec2_pad, vec0_pad)
    cross42 = torch.cross(vec4_pad, vec2_pad)
    cross64 = torch.cross(vec6_pad, vec4_pad)
    cross06 = torch.cross(vec0_pad, vec6_pad)

    # normalmap = cross20
    normalmap = F.normalize(cross20)
    normalmap += F.normalize(cross42)
    normalmap += F.normalize(cross64)
    normalmap += F.normalize(cross06)

    normalmap = F.normalize(normalmap)
    # scale the values from (-1, 1) to (0, 1)
    #normalmap = (-normalmap + 1) / 2
    reshape_normal = F.pad(normalmap, (1,1,1,1), 'constant', 0)

    n1 = reshape_normal[:, :, :h, 1:-1].contiguous().view(batch, 3, -1)
    n2 = reshape_normal[:, :, 2:, 1:-1].contiguous().view(batch, 3, -1)
    n3 = reshape_normal[:, :, 1:-1, :w].contiguous().view(batch, 3, -1)
    n4 = reshape_normal[:, :, 1:-1, 2:].contiguous().view(batch, 3, -1)
    cur = 1 - torch.einsum('bij,bij->bj', n1, n2)/2
    cur += 1 - torch.einsum('bij,bij->bj', n3, n4)/2

    cur /= 2
    cur[cur<1e-6] = 0
    cur = cur.view(batch, h, w)
    return normalmap#, cur

def dibr_vertical(depth, image, uvgrid, sgrid, baseline):
    #depth = remove_flying_pixel(depth)
    disp = torch.cat(
                (
                    torch.zeros_like(depth),
                    S360.derivatives.dtheta_vertical(sgrid, depth, baseline)
                ),
                dim=1
            )
    render_coords = uvgrid + disp
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    rendered,_ = L.splatting.render(image, depth, render_coords, max_depth=8)
    return rendered

def dibr_horizontal(depth, image, uvgrid, sgrid, baseline):
    #depth = remove_flying_pixel(depth)
    disp = torch.cat(
                (
                    S360.derivatives.dphi_horizontal_clip(sgrid, depth, baseline),
                    S360.derivatives.dtheta_horizontal_clip(sgrid, depth, baseline)
                ),
                dim=1
            )
    render_coords = uvgrid + disp
    render_coords[:, 0, :, :] = torch.fmod(render_coords[:, 0, :, :] + 512, 512)
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    rendered, _ = L.splatting.render(image, depth, render_coords, max_depth=8)
    return rendered    

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D    

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,h,w = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,h, w), grad_x.view(N,C,h, w)), dim=1)