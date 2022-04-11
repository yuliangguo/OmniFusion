   
import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import torch.nn.functional as F
import numpy as np
from equi_pers.equi2pers_v3 import equi2pers
from equi_pers.pers2equi_v3 import pers2equi 
import functools
import copy
from .blocks import Transformer_Block 


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU_v2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), 
                              stride=(stride, stride, 1), padding=(pad, pad, 0), bias=False, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
    
class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
    
        
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
    
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network
            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

   
def convert_conv(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.Conv2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.Conv3d(m.in_channels, m.out_channels, kernel_size=(m.kernel_size[0], m.kernel_size[1], 1), 
                                  stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0), padding_mode='zeros', bias=False)
                    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
                    if m.bias is not None:
                        new_layer.bias.data.copy_(m.bias.data)
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_conv(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def convert_bn(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.BatchNorm3d(m.num_features)
                    new_layer.weight.data.copy_(m.weight.data)
                    new_layer.bias.data.copy_(m.bias.data)
                    new_layer.running_mean.data.copy_(m.running_mean.data)
                    new_layer.running_var.data.copy_(m.running_var.data)
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_bn(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer            
    
class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)
            
        encoded = self.encoder_norm(hidden_states)
        #hooked = hooked[::-1]       
        return encoded       
       
        
class spherical_fusion(nn.Module):
    def __init__(self, nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80)):
        self.nrows = nrows
        self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        super(spherical_fusion, self).__init__()
        pretrain_model = torchvision.models.resnet34(pretrained=True)

        encoder = convert_conv(pretrain_model)
        encoder = convert_bn(encoder)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1

        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  #64
        self.layer2 = encoder.layer2  #128
        self.layer3 = encoder.layer3  #256
        self.layer4 = encoder.layer4  #512

        self.down = nn.Conv3d(512, 512//16, kernel_size=1, stride=1, padding=0)
        self.transformer = Transformer_cascade(512, npatches, depth=6, num_heads=4)
        
        self.de_conv0_0 = ConvBnReLU_v2(512, 256, kernel_size=3, stride=1)
        self.de_conv0_1 = ConvBnReLU_v2(256+256, 128, kernel_size=3, stride=1) 
        self.de_conv1_0 = ConvBnReLU_v2(128, 128, kernel_size=3, stride=1)
        self.de_conv1_1 = ConvBnReLU_v2(128+128, 64, kernel_size=3, stride=1)
        self.de_conv2_0 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv2_1 = ConvBnReLU_v2(64+64, 64, kernel_size=3, stride=1)
        self.de_conv3_0 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv3_1 = ConvBnReLU_v2(64+64, 32, kernel_size=3, stride=1)
        self.de_conv4_0 = ConvBnReLU_v2(32, 32, kernel_size=3, stride=1)
        self.pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.weight_pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.min_depth = 0.1
        self.max_depth = 8.0

        self.mlp_points = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
        )

    
    def forward(self, rgb, confidence=True):
        bs, _, erp_h, erp_w = rgb.shape
        device = rgb.device
        patch_h, patch_w = pair(self.patch_size)
        
        high_res_patch, _, _, _ = equi2pers(rgb, self.fov, self.nrows, patch_size=self.patch_size)
        _, xyz, uv, center_points = equi2pers(rgb, self.fov, self.nrows, patch_size=(patch_h//4, patch_w//4))
        rho = torch.ones((uv.shape[0], 1, patch_h//4, patch_w//4), dtype=torch.float32, device=device)

        center_points = center_points.to(device)
        center_points = center_points.reshape(-1, 2, 1, 1).repeat(1, 1, patch_h//4, patch_w//4)
        n_patch = high_res_patch.shape[-1]
        new_xyz = torch.cat([center_points, rho, center_points], 1)
        point_feat = self.mlp_points(new_xyz.contiguous())
        point_feat = point_feat.permute(1, 2, 3, 0).unsqueeze(0)
    
        conv1 = self.relu(self.bn1(self.conv1(high_res_patch)))
        pool = F.max_pool3d(conv1, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        layer1 = self.layer1(pool)
        layer1 = layer1 + point_feat
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_reshape = self.down(layer4)
        layer4_reshape = layer4_reshape.reshape(bs, -1, n_patch).transpose(1, 2)

        layer4_reshape = self.transformer(layer4_reshape)
        layer4_reshape = layer4_reshape.transpose(1, 2).reshape(bs, -1, 1, 1, n_patch)
        layer4 = layer4 + layer4_reshape
        
        layer4 = layer4.permute(0, 4, 1, 2, 3).reshape(bs*n_patch, -1, layer4.shape[-3], layer4.shape[-2])
        up = F.interpolate(layer4, size=(layer3.shape[-3], layer3.shape[-2]), mode='bilinear', align_corners=False)
        up = up.reshape(bs, n_patch, -1, layer3.shape[-3], layer3.shape[-2]).permute(0, 2, 3, 4, 1)

        de_conv0_0 = self.de_conv0_0(up)
        concat = torch.cat([de_conv0_0, layer3], 1)
        de_conv0_1 = self.de_conv0_1(concat)
         
        de_conv0_1 = de_conv0_1.permute(0, 4, 1, 2, 3).reshape(bs*n_patch, -1, de_conv0_1.shape[-3], de_conv0_1.shape[-2])
        up = F.interpolate(de_conv0_1, size=(layer2.shape[-3], layer2.shape[-2]), mode='bilinear', align_corners=False)
        up = up.reshape(bs, n_patch, -1, layer2.shape[-3], layer2.shape[-2]).permute(0, 2, 3, 4, 1)
        de_conv1_0 = self.de_conv1_0(up)
        concat = torch.cat([de_conv1_0, layer2], 1)
        de_conv1_1 = self.de_conv1_1(concat)
        
        de_conv1_1 = de_conv1_1.permute(0, 4, 1, 2, 3).reshape(bs*n_patch, -1, de_conv1_1.shape[-3], de_conv1_1.shape[-2])
        up = F.interpolate(de_conv1_1, size=(layer1.shape[-3], layer1.shape[-2]), mode='bilinear', align_corners=False)
        up = up.reshape(bs, n_patch, -1, layer1.shape[-3], layer1.shape[-2]).permute(0, 2, 3, 4, 1)
        de_conv2_0 = self.de_conv2_0(up)
        concat = torch.cat([de_conv2_0, layer1], 1)
        de_conv2_1 = self.de_conv2_1(concat)
        
        de_conv2_1 = de_conv2_1.permute(0, 4, 1, 2, 3).reshape(bs*n_patch, -1, de_conv2_1.shape[-3], de_conv2_1.shape[-2])
        up = F.interpolate(de_conv2_1, size=(conv1.shape[-3], conv1.shape[-2]), mode='bilinear', align_corners=False)
        up = up.reshape(bs, n_patch, -1, conv1.shape[-3], conv1.shape[-2]).permute(0, 2, 3, 4, 1)
        de_conv3_0 = self.de_conv3_0(up)
        concat = torch.cat([de_conv3_0, conv1], 1)
        de_conv3_1 = self.de_conv3_1(concat)  
         
        de_conv3_1 = de_conv3_1.permute(0, 4, 1, 2, 3).reshape(bs*n_patch, -1, de_conv3_1.shape[-3], de_conv3_1.shape[-2])     
        up = F.interpolate(de_conv3_1, (patch_h, patch_w), mode='bilinear')
        up = up.reshape(bs, n_patch, -1, patch_h, patch_w).permute(0, 2, 3, 4, 1)
        de_conv4_0 = self.de_conv4_0(up) 
        
        pred = F.relu(self.pred(de_conv4_0))
        if confidence:
            weight = torch.sigmoid(self.weight_pred(de_conv4_0))
            pred = pred * weight
            pred = pers2equi(pred, self.fov, self.nrows, (patch_h, patch_w), (erp_h, erp_w), 'pred')  
            weight = pers2equi(weight, self.fov, self.nrows, (patch_h, patch_w), (erp_h, erp_w), 'weight')      
            zero_weights = (weight <= 1e-8).detach().type(torch.float32)
            pred = pred / (weight + 1e-8 * zero_weights)
        else:
            pred = pers2equi(pred, self.fov, self.nrows, (patch_h, patch_w), (erp_h, erp_w), 'pred')  
        return pred
    
if __name__ == "__main__":
    net = spherical_fusion()   
    input = torch.zeros((1, 3, 128, 256), dtype=torch.float32)
    output = net(input, input, 86, (32, 32))
    print(output.shape)

        
            
            
        
        
               