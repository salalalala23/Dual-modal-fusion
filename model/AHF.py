import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import get_data
# from SJC import get_data    #改過
import torch.utils.data
from torch.utils.data import DataLoader
from libtiff import TIFF
import scipy.io as scio
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import time
# import torchsummary
from torch.optim import lr_scheduler


class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, expansion, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes*2))
        self.conv2 = nn.Conv2d(int(planes*2), int(planes*2), kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(int(planes*2))
        self.conv3 = nn.Conv2d(int(planes*2), 128, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes*2, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*8))


        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3,2,1))
        # 分类用的全连接
        a=int(patch_size/8)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*8 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        batch_size = y.shape[0]
        c = y.shape[1]
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f3 = f1.reshape(batch_size, c, S)
        f4 = f2.reshape(batch_size, c, S)
        # print('1',f1.shape,f2.shape)
        I_hat = (1. / S) * torch.eye(S, S, device=f1.device) + (-1. / S / S) * torch.ones(S, S,
                                                                                          device=f1.device)  # (1/S)*(I-(1/S)*(1))
        # print('a',I_hat.shape)
        I_hat = I_hat.view(1, S, S).repeat(batch_size, 1, 1).type(f1.dtype)  # 扩展成cxSxS大小
        # print(I_hat.shape)
        cov_m = f3.bmm(I_hat).bmm(f4.transpose(1, 2))
        # print(cov_m.type(),cov_m.shape)
        l_sum = cov_m.sum(1)  # 求列和
        h_sum = cov_m.sum(2)  # 求行和
        a1 = torch.sigmoid(l_sum)
        a2 = torch.sigmoid(h_sum)
        # print(a1)
        a1 = 1 - a1
        a2 = 1 - a2
        #
        a1 = a1.reshape(a1.shape[0], a1.shape[1], 1, 1)
        a2 = a2.reshape(a2.shape[0], a2.shape[1], 1, 1)
        # print(a1)
        f = f1 * a1 + f2 * a2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # b = b1+b2  # 16x16x192
        # b = torch.cat((b1, b2), 1)
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = self.Upsample_fusion(c1, c2)
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer2(b))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet18new(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18new, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes*2, init_planes*8, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(init_planes*8))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3,2,1))

        # 分类用的全连接
        a=int(patch_size/16)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*8 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(Bottleneck(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        batch_size = y.shape[0]
        c = y.shape[1]
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f3 = f1.reshape(batch_size, c, S)
        f4 = f2.reshape(batch_size, c, S)
        # print('1',f1.shape,f2.shape)
        I_hat = (1. / S) * torch.eye(S, S, device=f1.device) + (-1. / S / S) * torch.ones(S, S,
                                                                                          device=f1.device)  # (1/S)*(I-(1/S)*(1))
        # print('a',I_hat.shape)
        I_hat = I_hat.view(1, S, S).repeat(batch_size, 1, 1).type(f1.dtype)  # 扩展成cxSxS大小
        # print(I_hat.shape)
        cov_m = f3.bmm(I_hat).bmm(f4.transpose(1, 2))
        # print(cov_m.type(),cov_m.shape)
        l_sum = cov_m.sum(1)  # 求列和
        h_sum = cov_m.sum(2)  # 求行和
        a1 = torch.sigmoid(l_sum)
        a2 = torch.sigmoid(h_sum)
        # print(a1)
        a1 = 1 - a1
        a2 = 1 - a2
        #
        a1 = a1.reshape(a1.shape[0], a1.shape[1], 1, 1)
        a2 = a2.reshape(a2.shape[0], a2.shape[1], 1, 1)
        # print(a1)
        f = f1 * a1 + f2 * a2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # b = b1+b2  # 16x16x192
        # b = torch.cat((b1, b2), 1)
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = self.Upsample_fusion(c1, c2)
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer2(b))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet18add(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18add, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes*2, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*8))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes * 8, init_planes * 8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3, 2, 1))

        # 分类用的全连接
        a=int(patch_size/4)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*2 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)
    def Upsample_fusion(self, x, y):
        star_time = time.time()
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f = f1+f2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f
    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = b1+b2  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = c1+c2
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3 , self.lat_layer1(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2 , self.lat_layer2(b))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet18cat(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18cat, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*8, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes*4, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*8))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes * 8 * 3, init_planes * 8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3, 2, 1))

        # 分类用的全连接
        a=int(patch_size/8)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*8 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f = torch.cat((f1,f2),1)
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = torch.cat((b1, b2), 1)
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = torch.cat((c1, c2), 1)
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer2(b))
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet14(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet14, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
          # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        #self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*4, init_planes*4, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*2, init_planes*4, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes, init_planes*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*4))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes*4, init_planes*4, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3,2,1))

        # 分类用的全连接
        a=int(patch_size/4)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*4 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        batch_size = y.shape[0]
        c = y.shape[1]
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f3 = f1.reshape(batch_size, c, S)
        f4 = f2.reshape(batch_size, c, S)
        # print('1',f1.shape,f2.shape)
        I_hat = (1. / S) * torch.eye(S, S, device=f1.device) + (-1. / S / S) * torch.ones(S, S,
                                                                                          device=f1.device)  # (1/S)*(I-(1/S)*(1))
        # print('a',I_hat.shape)
        I_hat = I_hat.view(1, S, S).repeat(batch_size, 1, 1).type(f1.dtype)  # 扩展成cxSxS大小
        # print(I_hat.shape)
        cov_m = f3.bmm(I_hat).bmm(f4.transpose(1, 2))
        # print(cov_m.type(),cov_m.shape)
        l_sum = cov_m.sum(1)  # 求列和
        h_sum = cov_m.sum(2)  # 求行和
        a1 = torch.sigmoid(l_sum)
        a2 = torch.sigmoid(h_sum)
        # print(a1)
        a1 = 1 - a1
        a2 = 1 - a2
        a1 = a1.reshape(a1.shape[0], a1.shape[1], 1, 1)
        a2 = a2.reshape(a2.shape[0], a2.shape[1], 1, 1)
        # print(a1)
        f = f1 * a1 + f2 * a2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        a=self.Upsample_fusion(a1,a2)
        # print(a1.shape,a.shape)
        b1 = self.layer2(a1)  # 16x16x128
        b2 = self.layer2(a2)  # 16x16x128
        b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c = self.layer3(b)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(c)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer1(b))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer2(a))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet20(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet20, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层
        self.layer5 = self.make_layer(init_planes*8, init_planes*16, layers[4], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*16, init_planes*16, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*8, init_planes*16, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(init_planes*4, init_planes*16, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Sequential(
            nn.Conv2d(init_planes*2, init_planes*16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*16))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes*16, init_planes*16, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3,2,1))

        # 分类用的全连接
        a=int(patch_size/8)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*16 * a * a, 32),
            nn.Linear(32, 16),
            nn.Linear(16, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        batch_size = y.shape[0]
        c = y.shape[1]
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f3 = f1.reshape(batch_size, c, S)
        f4 = f2.reshape(batch_size, c, S)
        # print('1',f1.shape,f2.shape)
        I_hat = (1. / S) * torch.eye(S, S, device=f1.device) + (-1. / S / S) * torch.ones(S, S,
                                                                                          device=f1.device)  # (1/S)*(I-(1/S)*(1))
        # print('a',I_hat.shape)
        I_hat = I_hat.view(1, S, S).repeat(batch_size, 1, 1).type(f1.dtype)  # 扩展成cxSxS大小
        # print(I_hat.shape)
        cov_m = f3.bmm(I_hat).bmm(f4.transpose(1, 2))
        # print(cov_m.type(),cov_m.shape)
        l_sum = cov_m.sum(1)  # 求列和
        h_sum = cov_m.sum(2)  # 求行和
        a1 = torch.sigmoid(l_sum)
        a2 = torch.sigmoid(h_sum)
        # print(a1)
        a1 = 1 - a1
        a2 = 1 - a2
        a1 = a1.reshape(a1.shape[0], a1.shape[1], 1, 1)
        a2 = a2.reshape(a2.shape[0], a2.shape[1], 1, 1)
        # print(a1)
        f = f1 * a1 + f2 * a2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = self.Upsample_fusion(c1, c2)
        d = self.layer4(c)
        e = self.layer5(d)
        # print(c.shape)
        # Top-down
        f4 = self.toplayer(e)
        f3 = self.Upsample_fusion(f4, self.lat_layer1(d))  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer2(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer3(b))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        fe = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(fe)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x

#待完成
class ResNet18org(nn.Module):  # 224x224x3
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18org, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0],stride=2)  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1])  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1],stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2],stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3],stride=2)  # 融合层

        # # Top layer
        # self.toplayer = nn.Conv2d(init_planes*8, init_planes*2, kernel_size=1, stride=1, padding=0)
        # self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*2, kernel_size=1, stride=1, padding=0)
        # self.lat_layer2 = nn.Sequential(
        #     nn.Conv2d(init_planes*2, init_planes*2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(init_planes*2))
        #
        # # Smooth layer
        # self.smooth = nn.Conv2d(init_planes*2, init_planes*2, kernel_size=3, stride=1, padding=1)

        # 分类用的全连接
        a=int(patch_size/8)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*8, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        #b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        #c = self.Upsample_fusion(c1, c2)
        c=c1+c2
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        # f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        # f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        # f1 = self.Upsample_fusion(f2, self.lat_layer2(b))  # 16x16x288融合
        # # smooth
        # f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        d = F.avg_pool2d(d, d.shape[2])  # 1x1x512
        e = d.view(d.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x

#待完成
class ResNet18AW(nn.Module):  # patchMS=patchPAN
    # 实现主module:ResNet14
    def __init__(self,layers, num_classes, patch_size,init_planes):
        super(ResNet18AW, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1],stride=2)  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[2], stride=2)  # 融合层

        # Top layer
        # self.toplayer = nn.Conv2d(init_planes*8, init_planes*2, kernel_size=1, stride=1, padding=0)
        # self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*2, kernel_size=1, stride=1, padding=0)
        # self.lat_layer2 = nn.Sequential(
        #     nn.Conv2d(init_planes*2, init_planes*2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(init_planes*2))

        # Smooth layer
        # self.smooth = nn.Conv2d(init_planes*2, init_planes*2, kernel_size=3, stride=1, padding=1)

        # 分类用的全连接
        a=int(patch_size/16)
        self.fc = nn.Sequential(
            nn.Linear(init_planes*8 * a * a, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

   

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        # b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        # c = self.Upsample_fusion(c1, c2)
        c=c1+c2
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        # f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        # f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        # f1 = self.Upsample_fusion(f2, self.lat_layer2(b))  # 16x16x288融合
        # # smooth
        # f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = d.view(d.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


class ResNet18CAFF(nn.Module):  # 224x224x3
    # 实现主module:ResNet18 patch:32x32 128x128
    def __init__(self, args):
        super(ResNet18CAFF, self).__init__()  # ((input-kernel_size+2*padding)/stride)+1
        layers = [2, 2, 2, 2]
        num_classes = 10
        patch_size = 16
        init_planes = 16
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, init_planes, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(init_planes),  # 64x64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1((64+2*1-3)/2)+1=32
        )  # 32x32x64

        # 重复的layer,分别有2,2,2个residual block
        # Bottom-up layers
        # 第一个stride=2，剩下1个stride=1;
        self.layer1_1 = self.make_layer(init_planes, init_planes, layers[0])  # 非共享
        self.layer1_2 = self.make_layer(init_planes, init_planes, layers[0],stride=2)  # 非共享
        self.layer2_1 = self.make_layer(init_planes, init_planes*2, layers[1])  # 非共享层
        self.layer2_2 = self.make_layer(init_planes, init_planes*2, layers[1], stride=2)  # 非共享层
        self.layer3 = self.make_layer(init_planes*2, init_planes*4, layers[2], stride=2)  # 共享层
        self.layer4 = self.make_layer(init_planes*4, init_planes*8, layers[3], stride=2)  # 融合层

        # Top layer
        self.toplayer = nn.Conv2d(init_planes*8, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(init_planes*4, init_planes*8, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Sequential(
            nn.Conv2d(init_planes*2, init_planes*8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_planes*8))

        # Smooth layer
        self.smooth = nn.Sequential(
            nn.Conv2d(init_planes * 8, init_planes * 8, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(3, 2, 1))
        # 分类用的全连接
        a = int(patch_size / 4)
        self.fc = nn.Sequential(
            nn.Linear(init_planes * 8 * a * a, args.Categories))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def Upsample_fusion(self, x, y):
        star_time = time.time()
        batch_size = y.shape[0]
        c = y.shape[1]
        h = y.shape[2]
        hx = x.shape[2]
        w = y.shape[3]
        S = h * w
        if hx < h:
            f1 = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        else:
            f1 = x
        f2 = y
        f3 = f1.reshape(batch_size, c, S)
        f4 = f2.reshape(batch_size, c, S)
        # print('1',f1.shape,f2.shape)
        I_hat = (1. / S) * torch.eye(S, S, device=f1.device) + (-1. / S / S) * torch.ones(S, S,
                                                                                          device=f1.device)  # (1/S)*(I-(1/S)*(1))
        # print('a',I_hat.shape)
        I_hat = I_hat.view(1, S, S).repeat(batch_size, 1, 1).type(f1.dtype)  # 扩展成cxSxS大小
        # print(I_hat.shape)
        cov_m = f3.bmm(I_hat).bmm(f4.transpose(1, 2))
        # print(cov_m.type(),cov_m.shape)
        l_sum = cov_m.sum(1)  # 求列和
        h_sum = cov_m.sum(2)  # 求行和
        a1 = torch.sigmoid(l_sum)
        a2 = torch.sigmoid(h_sum)
        # print(a1)
        a1 = 1 - a1
        a2 = 1 - a2
        a1 = a1.reshape(a1.shape[0], a1.shape[1], 1, 1)
        a2 = a2.reshape(a2.shape[0], a2.shape[1], 1, 1)
        # print(a1)
        f = f1 * a1 + f2 * a2
        # print(f.shape)
        end_time = time.time()
        # print("fusion_time is %.2f s" % (end_time-star_time))
        return f

    def forward(self, x1, x2):  # 224x224x3
        ##Bottom-up
        x11 = self.pre_ms(x1)  # 32x32x64
        x22 = self.pre_pan(x2)  # 32x32x64
        # print(x2.shape)
        a1 = self.layer1_1(x11)  # 32x32x64
        a2 = self.layer1_2(x22)  # 32x32x64
        # print(a1.shape,a.shape)
        b1 = self.layer2_1(a1)  # 16x16x128
        b2 = self.layer2_2(a2)  # 16x16x128
        b = self.Upsample_fusion(b1, b2)  # 16x16x192
        # print(b2.shape,b.shape)
        # x = torch.cat((b1, b2), 1)# 8x8x512
        c1 = self.layer3(b1)  # 8x8x25
        c2 = self.layer3(b2)  # 8x8x25
        c = self.Upsample_fusion(c1, c2)
        d = self.layer4(c)
        # print(c.shape)
        # Top-down
        f3 = self.toplayer(d)  # 8x8x512 ==>  8x8x384 降通道
        f2 = self.Upsample_fusion(f3, self.lat_layer1(c))  # 8x8x192 融合
        f1 = self.Upsample_fusion(f2, self.lat_layer2(b))  # 16x16x288融合
        # smooth
        f1 = self.smooth(f1)
        # print(f1.shape)        #  分类
        # e = F.avg_pool2d(f1, f1.shape[2])  # 1x1x288
        # print(c.shape)
        e = f1.view(f1.size(0), -1)  # 将输出拉伸为一行：1x288
        # print(e.shape)
        x = self.fc(e)  # 1x1
        # print(x.shape)
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        # return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间
        return x


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    pan = torch.randn(2, 1, 64, 64)
    pan = pan.cuda()
    ms = torch.randn(2, 4, 16, 16)
    ms = ms.cuda()
    grf_net = ResNet18CAFF().cuda()
    # out_result,coefxy = grf_net(ms,pan)
    # out_result,SSIM = grf_net(ms,pan)
    # 输入为MS和PAN
    # torchsummary.summary(grf_net, input_size=[(4, 16, 16), (1, 64, 64)])
    out_result = grf_net(ms, pan)
    print(out_result)
    print(out_result.shape)