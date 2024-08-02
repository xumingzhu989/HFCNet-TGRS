from tkinter.tix import Tree
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
# import scipy.stats
from torch.utils.data import Dataset
import cv2
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import model.vgg as vgg
from swintransformer.backbone import SwinTransformer
from typing import Any, Optional, Tuple, Type
from model.ASPP import  ASPP_v4

def segm_swin(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=0.0, frozen_stages=-1
                                         )
    if pretrained:
        print('Initializing Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Swin Transformer weights.')
        backbone.init_weights()
    return backbone


# 以下是完整模型
class HFCNet(nn.Module):  
    def __init__(self, args):
        super(HFCNet,self).__init__()
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.sig = nn.Sigmoid()
        ################################################################################################################################
        # VGG backbone
        self.vgg = vgg.VGG('rgb')
        self.layer1 = self.vgg.conv1
        self.layer2 = self.vgg.conv2
        self.layer3 = self.vgg.conv3
        self.layer4 = self.vgg.conv4
        self.layer5 = self.vgg.conv5
        ################################################################################################################################
        # swinb backbone
        self.swin = segm_swin(args.pretrained, args)
        ################################################################################################################################
        
        self.decoder = SGAED()
        self.pos = PositionEmbeddingRandom(64, 1.0)

        self.fusion1 = AGLI(256,256)
        self.fusion2 = AGLI(512,256)
        self.fusion3 = AGLI(1024,256)

        self.conv1 = nn.Sequential(BasicConv2d(1024 + 256, 256, 3, padding=1), BasicConv2d(256, 256, 3, padding=1))
        self.conv2 = nn.Sequential(BasicConv2d(512 + 256, 256, 3, padding=1), BasicConv2d(256, 256, 3, padding=1))
        self.conv3 = nn.Sequential(BasicConv2d(256 + 256, 256, 3, padding=1), BasicConv2d(256, 256, 3, padding=1))
        self.aspp = ASPP_v4(256, [3, 6, 9])
    def forward(self, Xin):
        xC1 = self.layer1(Xin)  # 64*224*224
        xC2 = self.layer2(xC1)  # 128*112*112
        xC3 = self.layer3(xC2) # 256*56*56
        xs1 = self.swin(xC2,0,1) # 256*56*56
        xS1 = self.swin(xs1,1,0) #256*56*56
        att1,xS1,xC3,f_s1,f_c3= self.fusion1(xS1,xC3)

        xC4 = self.layer4(xC3)
        xs2 = self.swin(xS1,1,1) #512*28*28
        xS2 = self.swin(xs2,2,0) #512*28*28
        att2,xS2,xC4,f_s2,f_c4= self.fusion2(xS2,xC4)

        xC5 = self.layer5(xC4)
        xs3 = self.swin(xS2,2,1) #1024*14*14
        xS3 = self.swin(xs3,3,0) #1024*14*14
        att3,xS3,xC5,f_s3,f_c5= self.fusion3(xS3,xC5)

        xS3 = self.conv1(torch.cat((xS3,xC5),dim=1))
        xS2 = self.conv2(torch.cat((xS2,xC4),dim=1))
        xS1 = self.conv3(torch.cat((xS1,xC3),dim=1))
        X_l = self.aspp(xS3)
        
        s1, s2, s3, s4, s5, s6= self.decoder(X_l,xS3, xS2, xS1, xC2, xC1)

        s1 = self.sig(s1.squeeze(dim=1))
        s2 = self.sig(self.upsample2(s2).squeeze(dim=1))
        s3 = self.sig(self.upsample4(s3).squeeze(dim=1))
        s4 = self.sig(self.upsample8(s4).squeeze(dim=1))
        s5 = self.sig(self.upsample16(s5).squeeze(dim=1))
        s6 = self.sig(self.upsample16(s6).squeeze(dim=1))

        outs = [s1, s2, s3, s4, s5, s6]
        atts = [att1,att2,att3]
        fs = [f_s1,f_s2,f_s3]
        fc = [f_c3,f_c4,f_c5]
        return outs, atts , fs, fc



class AGLI(nn.Module):
    def __init__(self,ch1,ch2):
        super(AGLI, self).__init__()
        self.conv1 = BasicConv2d(ch1, 32, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(ch2, 32, kernel_size=3, padding=1)
        self.convg1 = BasicConv2d(ch1, 32, kernel_size=3, padding=1)
        self.convg2 = BasicConv2d(ch2, 32, kernel_size=3, padding=1)
        self.out1 = BasicConv2d(ch1, ch1, kernel_size=3, padding=1)
        self.out2 = BasicConv2d(ch2, ch2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.linear_s = nn.Sequential(
            nn.Conv2d(ch1+32, ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True)
        )
        self.linear_c = nn.Sequential(
            nn.Conv2d(ch2+32, ch2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True)
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.out_att = BasicConv2d(1, 1, kernel_size=3, padding=1)

    def forward(self,xs,xc):
        g_s = self.convg1(xs)
        g_c = self.convg2(xc)
        g_s = self.GAP(g_s)
        g_c = self.GAP(g_c) # b,256,1,1
        gs = F.interpolate(g_s, size=xs.shape[-2:], mode='bilinear', align_corners=False)
        gc = F.interpolate(g_c, size=xc.shape[-2:], mode='bilinear', align_corners=False)
        xs = self.linear_s(torch.cat((gs,xs),dim=1)) + xs
        xc = self.linear_c(torch.cat((gc,xc),dim=1)) + xc

        f_s = self.conv1(xs) #b,256,h,w
        f_c = self.conv2(xc)
        fs = f_s.view(f_s.size(0),f_s.size(1),-1)  # b,32,hw
        fc = f_c.view(f_c.size(0),f_c.size(1),-1) # b,32,hw
        att_map = fs.permute(0,2,1) @ fc # b,hw,hw
        outs = (xs.view(xs.size(0),xs.size(1),-1) @ att_map.softmax(-2)).reshape(xs.shape)
        outc = (xc.view(xc.size(0),xc.size(1),-1) @ att_map.transpose(-1,-2).softmax(-2)).reshape(xc.shape)
        out_s = self.out1(outs) + xs
        out_c = self.out2(outc) + xc
        att_map = att_map + att_map.transpose(-1,-2)
        att_map = att_map.softmax(-2)+att_map.transpose(-1,-2).softmax(-2)
        att_map = self.out_att(att_map.unsqueeze(1))
        return att_map,out_s,out_c, g_s, g_c



class SGAED(nn.Module):
    def __init__(self):
        super(SGAED, self).__init__()

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample16 = nn.Upsample(scale_factor=0.0625, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.decoder6 = CalculateUnitAE(256, 0, 256, 256)
        self.S6 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder5 = CalculateUnitAE(256, 256, 256, 256)
        self.S5 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder4 = CalculateUnitAE(256, 256, 256, 256)
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = CalculateUnitAE(256, 256, 128, 128)
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = CalculateUnitAE(128, 128, 64, 64)
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = CalculateUnitAE(64, 64, 64, 32)
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x6, x5, x4, x3, x2, x1):

        x6 = self.decoder6(x6, 0, 0)  + x6 
        s6 = self.S6(x6)  

        x5 = self.decoder5(x5, x6, s6)  
        x5_up = self.upsample2(x5)
        s5 = self.S5(x5) 

        x4 = self.decoder4(x4, x5_up, self.upsample2(s5))  
        x4_up = self.upsample2(x4)
        s4 = self.S4(x4) 

        x3 = self.decoder3(x3, x4_up, self.upsample2(s4)) 
        x3_up = self.upsample2(x3)
        s3 = self.S3(x3)

        x2 = self.decoder2(x2, x3_up, self.upsample2(s3))  
        x2_up = self.upsample2(x2)
        s2 = self.S2(x2)

        x1 = self.decoder1(x1, x2_up, self.upsample2(s2))  
        s1 = self.S1(x1)

        return s1, s2, s3, s4, s5, s6


class CalculateUnitAE(nn.Module):
    def __init__(self, in_Channel1=0, in_Channel2=0, mid_Channel3=0, out_Channel=0):  # 四个个输入（从in1开始有几个填几个，没有这一个就填0，要保证每个输入的HW一样，最后一个用来生成空间注意力图）
        super(CalculateUnitAE, self).__init__()
        self.sig = nn.Sigmoid()
        self.AE1 = AE(in_Channel1)
        self.AE2 = AE(in_Channel2)
        self.conv = nn.Sequential(BasicConv2d(in_Channel1 + in_Channel2, mid_Channel3, 3, padding=1), BasicConv2d(mid_Channel3, out_Channel, 3, padding=1))  # 这里是每个小模块用到的卷积，可以改成其他的

    def forward(self, in1, in2=0, inSA=0):
        if (torch.is_tensor(inSA)):
            SA = self.sig(inSA)
        else:
            SA = 0
        x = self.AE1(in1, SA)
        if (torch.is_tensor(in2)):
            in2 = self.AE2(in2, SA)
            x = torch.cat([x, in2], dim=1)
        out = self.conv(x)
        return out


class AE(nn.Module):
    def __init__(self, ch):  #
        super(AE, self).__init__()
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(ch)

    def forward(self, inx, inSA=0):
        x = self.CA(inx, inSA)
        # x = self.CA(inx)
        out = self.SA(x, inSA) * x + x#inx
        return out

    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, inSA=0):
        out = 0
        if torch.is_tensor(inSA):
            inSA = inSA.view(inSA.size(0),inSA.size(1),-1).permute(0,2,1)
            f = x.view(x.size(0),x.size(1),-1)  
            ca = torch.matmul(f,inSA).unsqueeze(-1)  # (B,C,1,1)
            ca = self.fc2(self.relu1(self.fc1(ca)))
            out = out + ca
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = out + max_out
        xout = self.sigmoid(out) * x + x
        return  xout
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inSA=0):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        if torch.is_tensor(inSA):
            x = torch.cat(( max_out, inSA), dim=1)
            x = self.conv1(x)
        else:
            # x = torch.cat((avg_out, max_out), dim=1)
            x = self.conv2(max_out)
        return self.sigmoid(x)

    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )


    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C



def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()