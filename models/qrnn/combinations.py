import torch
import torch.nn as nn


class ReLUConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(ReLUConv3d, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))


class ReLUDeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(ReLUDeConv3d, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))


class ReLUUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1,2,2), inplace=True):
        super(ReLUUpsampleConv3d, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class UpsampleConv3d(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)
        return out


class Conv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False):
        super(Conv3d, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class DeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False):
        super(DeConv3d, self).__init__()       
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=bias))


class BNUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1,2,2)):
        super(BNUpsampleConv3d, self).__init__()
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
