import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from block import *
import functools
import numpy as np


class ResBlock1(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """前向传播过程"""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, size, strides):
        super(ResBlock2, self).__init__()
        sequence = [nn.Conv2d(input_dim, output_dim, size, stride=strides, padding=int((size - 1) / 2))
            , nn.BatchNorm2d(output_dim)
            , nn.LeakyReLU(inplace=True)]

        self.module = nn.Sequential(*sequence)

    def forward(self, x):
        return self.module(x)

class Generator(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x
        xa = self.resBlock(x)

        ya = self.SAM(xa)

        x1 = self.down(xa)
        x1 = self.CMSM2(x1)
        y1 = self.SAM(x1)

        x2 = self.down1(x1)
        x2 = self.CMSM3(x2)
        y2 = self.SAM(x2)
        x3 = self.up1(x2)
        x4 = x3 + y1
        x4 = self.CMSM2(x4)
        y4 = self.SAM(x4)
        x5 = self.down1(x4)
        x5 = self.CMSM3(x5)
        x6 = x5 + y2
        x6 = self.CMSM3(x6)
        x7 = self.up1(x6)

        x_all = y1 + y4 + x7
        out = self.CMSM4(x_all)
        out = self.up2(out)
        out = out + ya
        out = self.CMSM1(out)
        out = self.cov(out)
        return out + res


class Generator1(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator1, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x
        xa = self.resBlock(x)

        ya = xa

        x1 = self.down(xa)
        x1 = self.CMSM2(x1)
        y1 = x1

        x2 = self.down1(x1)
        x2 = self.CMSM3(x2)
        y2 = x2
        x3 = self.up1(x2)
        x4 = x3 + y1
        x4 = self.CMSM2(x4)
        y4 = x4
        x5 = self.down1(x4)
        x5 = self.CMSM3(x5)
        x6 = x5 + y2
        x6 = self.CMSM3(x6)
        x7 = self.up1(x6)

        x_all = y1 + y4 + x7
        out = self.CMSM4(x_all)
        out = self.up2(out)
        out = out + ya
        out = self.CMSM1(out)
        out = self.cov(out)
        return out + res

class Generator2(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator2, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x
        xa = self.resBlock(x)

        ya = self.SAM(xa)

        x1 = self.down(xa)
        x1 = self.CMSM2(x1)
        y1 = self.SAM(x1)


        x_all = y1 + x1
        out = self.CMSM4(x_all)
        out = self.up2(out)
        out = out + ya
        out = self.CMSM1(out)
        out = self.cov(out)
        return out + res



class Generator3(nn.Module):
    """生成模型(4x)"""

    def __init__(self):
        """初始化模型配置"""
        super(Generator3, self).__init__()
        # 卷积模块1
        self.rsblock1 = ResBlock1(1, 64)

        self.CMSM1 = group_conv(64)

        self.down = ResBlock2(64, 128, 1, 2)

        self.CMSM2 = group_conv(128)
        self.down1 = ResBlock2(128, 256, 1, 2)

        self.CMSM3 = group_conv(256)
        self.up1 = Upsampler(256, 128)

        self.CMSM4 = group_conv(128)
        self.up2 = Upsampler(128, 64)
        self.CMSM5 = group_conv(64)
        self.cov = conv3x3(64, 1, 1)
        self.SAM = SpatialAttention()

        self.resBlock = self._makeLayer_(CBLC1, 1, 64, 1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        res = x
        xa = self.resBlock(x)

        ya = self.SAM(xa)

        x1 = self.down(xa)
        y1 = self.SAM(x1)

        x2 = self.down1(x1)
        y2 = self.SAM(x2)
        x3 = self.up1(x2)
        x4 = x3 + y1
        y4 = self.SAM(x4)
        x5 = self.down1(x4)
        x6 = x5 + y2
        x7 = self.up1(x6)

        x_all = y1 + y4 + x7
        out = self.up2(x_all)
        out = out + ya
        out = self.cov(out)
        return out + res







def test():
    y = torch.randn(1, 1, 256, 256)
    net = Generator3()
    x = net(y)
    print(x.shape)

if __name__ == "__main__":
    test()






























class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator. """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator.

        When n_layers = 3, this ia 70x70 PatchGAN

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGAN, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)