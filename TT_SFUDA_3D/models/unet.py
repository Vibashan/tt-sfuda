import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import pdb
from base_networks import *

# adapt from https://github.com/MIC-DKFZ/BraTS2017

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='bn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        # pdb.set_trace()
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        # pdb.set_trace()
        if y.shape[4]==18:
            y = F.pad(y,(0,1), "constant", 0)
        if y.shape[4]==76:
            y = F.pad(y,(0,1), "constant", 0)
        if y.shape[4]==154:
            y = F.pad(y,(0,1), "constant", 0)

        y = self.relu(self.bn2(self.conv2(y)))
        # if prev.shape[4]==19:
        #     prev = prev[:,:,:,:,0:18]

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y

class ConvD_wo(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='bn', first=False):
        super(ConvD_wo, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        # self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        # self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        # self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.conv1(x)
        y = self.relu(self.conv2(x))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.conv3(x)
        return self.relu(x + y)


class ConvU_wo(nn.Module):
    def __init__(self, planes, norm='bn', first=False):
        super(ConvU_wo, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            # self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        # self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        # self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        # pdb.set_trace()
        if not self.first:
            x = self.relu(self.conv1(x))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        # pdb.set_trace()
        if y.shape[4]==18:
            y = F.pad(y,(0,1), "constant", 0)
        if y.shape[4]==76:
            y = F.pad(y,(0,1), "constant", 0)
        if y.shape[4]==154:
            y = F.pad(y,(0,1), "constant", 0)

        y = self.relu(self.conv2(y))
        # if prev.shape[4]==19:
        #     prev = prev[:,:,:,:,0:18]

        y = torch.cat([prev, y], 1)
        y = self.relu(self.conv3(y))

        return y

class Unet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='bn', num_classes=5):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(1,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)
        
        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        # y3 = self.seg3(y3)
        # y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) #+ self.upsample(y2)

        return y1


class Unet_wo(nn.Module):
    def __init__(self, c=4, n=4, dropout=0.5, norm='bn', num_classes=5):
        super(Unet_wo, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD_wo(1,     n, dropout, norm, first=True)
        self.convd2 = ConvD_wo(n,   2*n, dropout, norm)
        self.convd3 = ConvD_wo(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD_wo(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD_wo(8*n,16*n, dropout, norm)

        self.convu4 = ConvU_wo(16*n, norm, True)
        self.convu3 = ConvU_wo(8*n, norm)
        self.convu2 = ConvU_wo(4*n, norm)
        self.convu1 = ConvU_wo(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode=None):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)
        
        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        # y3 = self.seg3(y3)
        # y2 = self.seg2(y2) + self.upsample(y3)
         #+ self.upsample(y2)
        if mode == 'const':
            y1 = self.seg1(y1)
            return y1, [x4,x5]
        else:
            y1 = self.seg1(y1)
            return y1


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
    ) -> None:
        super().__init__()
        
        norm_layer = nn.BatchNorm2d # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, 3)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class priorunet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(4, 4)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], input_channels, kernel_size=1)


    def forward(self, input):
        # pdb.set_trace()
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # output = self.final(x0_4)
        # pdb.set_trace()
        return x4_0

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.BatchNorm3d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        
        style = torch.flatten(style, 1)
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        gamma = gamma.unsqueeze(3)
        beta = beta.unsqueeze(3)
        # pdb.set_trace()
        out = gamma * out + beta

        return out
    

class adaptiveunet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='bn', num_classes=5):
        super(adaptiveunet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(1,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.adain1_e = AdaptiveInstanceNorm(n, 512)
        self.adain2_e = AdaptiveInstanceNorm(2*n, 512)
        self.adain3_e = AdaptiveInstanceNorm(4*n, 512)
        self.adain4_e = AdaptiveInstanceNorm(8*n, 512)

        self.adain1_d = AdaptiveInstanceNorm(16*n, 512)
        self.adain2_d = AdaptiveInstanceNorm(8*n, 512)
        self.adain3_d = AdaptiveInstanceNorm(4*n, 512)
        self.adain4_d = AdaptiveInstanceNorm(2*n, 512)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, prior):
        x1 = self.convd1(x)
        t = self.adain1_e(x1,prior)
        x2 = self.convd2(t)
        t = self.adain2_e(x2,prior)
        x3 = self.convd3(t)
        t = self.adain3_e(x3,prior)
        x4 = self.convd4(t)
        t = self.adain4_e(x4,prior)
        x5 = self.convd5(t)
        
        y4 = self.convu4(x5, x4)
        y4 = self.adain1_d(y4,prior)
        y3 = self.convu3(y4, x3)
        y3 = self.adain2_d(y3,prior)
        y2 = self.convu2(y3, x2)
        y2 = self.adain3_d(y2,prior)
        y1 = self.convu1(y2, x1)
        # y1 = self.adain4_d(y1,prior)

        # y3 = self.seg3(y3)
        # y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) #+ self.upsample(y2)

        return y1