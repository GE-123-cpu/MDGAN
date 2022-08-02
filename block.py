import torch.nn as nn
import torch
from torch.nn.modules.conv import Conv2d
import torch
from torch.autograd import Variable


class ResBlock(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(outChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(outChannals, outChannals, kernel_size=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        """前向传播过程"""
        resudial = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(x)
        out += resudial
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class Conv_block(nn.Module):
    def __init__(self, input_dim, output_dim, size, strides, num):
        super(Conv_block, self).__init__()
        sequence = [nn.Conv2d(input_dim, output_dim, size, stride=strides, padding=int((size - 1) / 2))
            , nn.BatchNorm2d(output_dim)
            , nn.LeakyReLU(inplace=True)]
        for _ in range(num):
            sequence.append(group_conv(output_dim))
        self.module = nn.Sequential(*sequence)

    def forward(self, x):
        return self.module(x)


class Residual_Block(nn.Module):
    def __init__(self, in_nc, mid_nc, kernel_size=3, stride=1, padding=1, neg_slope=0.2, res_scale=0.2):
        super(Residual_Block, self).__init__()
        self.res_scale = res_scale
        # block1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=mid_nc, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 1, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 2, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 3, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 4, out_channels=in_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding))

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(torch.cat((x, x1), 1))
        x3 = self.conv_block3(torch.cat((x, x1, x2), 1))
        x4 = self.conv_block4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv_block5(torch.cat((x, x1, x2, x3, x4), 1))
        # out = x5
        out = x5.mul(self.res_scale)
        return out


class group_conv(nn.Module):
    def __init__(self, dim, attention=False):
        super(group_conv, self).__init__()
        self.module1 = Res2NetBottleneck2(dim, dim)
        self.module2 = Res2NetBottleneck2(dim, dim)
        self.module3 = Res2NetBottleneck2(dim, dim)
        self.conv = nn.Conv2d(3 * dim, dim, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.cbam = nn.Sequential(ChannelAttention(3 * dim))
        self.attention = attention

    def forward(self, x):
        y1 = self.module1(x)
        y2 = self.module2(y1)
        y3 = self.module3(y2)
        y = torch.cat((y1, y2, y3), dim=1)
        if self.attention:
            y = self.cbam(y)
        y = self.bn(self.conv(y))

        return self.relu(x + y)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class Res2NetBottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, se=False, norm_layer=None, shrink=False):
        super(Res2NetBottleneck2, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = planes
        self.conv2 = nn.ModuleList(
            [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(bottleneck_planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.shrink = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.bn4 = norm_layer(planes)
        self.skip = shrink

    def forward(self, x):
        identity = x

        xs = torch.chunk(x, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            identity = self.bn4(self.shrink(identity))

        out += identity
        out = self.relu(out)

        return out


class CBLC(nn.Module):
    def __init__(self, inChannals, outChannals):
        super(CBLC, self).__init__()
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.block1 = group_conv(outChannals)

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block1(out)
        return out


class CBLC1(nn.Module):
    def __init__(self, inChannals, outChannals):
        super(CBLC1, self).__init__()
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.block1 = group_conv(outChannals)

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block1(out)
        return out


class CBLC2(nn.Module):
    def __init__(self, inChannals, outChannals):
        super(CBLC, self).__init__()
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.ConvTranspose2d(inChannals, outChannals, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class RRBA(nn.Module):
    def __init__(self, in_nc, mid_nc):
        super(RRBA, self).__init__()
        self.block1 = Residual_Block(in_nc, mid_nc)
        self.block2 = ChannelAttention(in_nc)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        return out1 + out2 + x


class Upsampler(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Upsampler, self).__init__()
        self.Res_Block = nn.Sequential(nn.Upsample(scale_factor=2)
                                       , Conv2d(input_dim, output_dim, 1, 1)
                                       , nn.BatchNorm2d(output_dim)
                                       , nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.Res_Block(x)


class RDB(nn.Module):
    def __init__(self, in_nc, mid_nc, kernel_size=3, stride=1, padding=1, neg_slope=0.2, res_scale=0.2):
        super(RDB, self).__init__()
        self.res_scale = res_scale
        # block1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=mid_nc, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 1, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 2, out_channels=in_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 3, out_channels=in_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        # block5
        self.conv_block5 = conv1x1(in_nc, in_nc)
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 5, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 6, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=False))
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + mid_nc * 7, out_channels=in_nc, kernel_size=kernel_size, stride=stride,
                      padding=padding))

        self.cov = conv1x1(in_nc, in_nc)
        self.bn = nn.BatchNorm2d(in_nc)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(torch.cat((x, x1), 1))
        x3 = self.conv_block3(torch.cat((x, x1, x2), 1))
        # x4 = self.conv_block4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.bn(self.conv_block5(x3))
        # out = self.relu(x5)
        # out = self.cov(x5)
        # out = x5.mul(self.res_scale)
        return self.relu(x5 + x)


class GRDB(nn.Module):
    def __init__(self, dim, attention=False):
        super(GRDB, self).__init__()
        self.module1 = RDB(dim, dim)
        self.module2 = RDB(dim, dim)
        self.module3 = RDB(dim, dim)
        self.module4 = RDB(dim, dim)
        self.conv = nn.Conv2d(3 * dim, dim, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.cbam = nn.Sequential(ChannelAttention(3 * dim))
        self.attention = attention
        self.conv1 = conv1x1(3 * dim, dim)

    def forward(self, x):
        y1 = self.module1(x)
        y2 = self.module2(y1)
        y3 = self.module3(y2)
        #y4 = self.module4(y3)
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.conv1(y)
        # y = self.conv1(y)
        # y = self.bn(y)
        # y = self.relu(y)
        # if self.attention:
        #    y = self.cbam(y)
        # y = self.bn(self.conv(y))

        return self.relu(x + y)


class CBAM(nn.Module):
    def __init__(self, in_nc, mid_nc):
        super(CBAM, self).__init__()
        self.block1 = SpatialAttention()
        self.block2 = ChannelAttention(in_nc)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.block2(x)
        out2 = self.block1(out1)
        return out2