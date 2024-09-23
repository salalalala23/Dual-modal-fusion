import torch
import torch.nn as nn
import torch.nn.functional as F
# import second.model.CT
import numpy as np
from sympy.matrices import Matrix, GramSchmidt
device = "cuda" if torch.cuda.is_available() else "cpu" 


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# get BasicBlock which layers < 50(18, 34)
class BasicBlk(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlk, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion * out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_ch)
            )
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:  # is not None
            x = self.downsample(x)  # resize the channel
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# get BottleBlock which layers >= 50
class BottleNck(nn.Module):
    expansion = 4  # the factor of the last layer of BottleBlock and the first layer of it

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BottleNck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, int(out_ch/4), kernel_size=1, stride=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(int(out_ch/4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(int(out_ch/4), int(out_ch/4), stride)
        self.bn2 = nn.BatchNorm2d(int(out_ch/4))
        self.conv3 = nn.Conv2d(int(out_ch/4), out_ch, kernel_size=1, stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        print(out.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(4, 64)
        self.conv2 = conv3x3(5, 64)
        self.conv3 = conv3x3(1, 64)
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, y=0, mode='1'):
        if mode == '2':
            # out = torch.concat([F.interpolate(x, size=(64, 64)), y], dim=1)
            out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
            out = self.relu(self.BN(self.conv2(out)))  # torch.Size([20, 64, 16, 16])
        else:
            upsample = nn.Upsample(scale_factor=4, mode='bilinear')
            out = upsample(x)
            out = self.relu(self.BN(self.conv1(out)))  # torch.Size([20, 64, 16, 16])
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        # visualize_channels(out, 8, 4, 'out1')
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        # visualize_channels(out, 8, 4, 'out2')
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        # visualize_channels(out, 8, 4, 'out3')
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        # visualize_channels(out, 8, 4, 'out4')
        out = F.avg_pool2d(out, 8)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out
    

class Resnet_M(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Resnet_M, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(4, 64)
        self.conv2 = conv3x3(5, 64)
        self.conv3 = conv3x3(1, 64)
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=0, mode='2'):
        if mode == '2':
            out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
            out = self.relu(self.BN(self.conv2(out)))  # torch.Size([20, 64, 16, 16])
        else:
            out = self.relu(self.BN(self.conv1(x)))  # torch.Size([20, 64, 16, 16])
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out1 = F.avg_pool2d(out, 16)
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out2 = F.avg_pool2d(out, 8)
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out3 = F.avg_pool2d(out, 4)
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out1 + out2 + out3 + out
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class Resnet_Base(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(Resnet_Base, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(4, 64)
        self.conv2 = conv3x3(5, 64)
        self.conv3 = conv3x3(1, 64)
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.in_planes = 64
        self.layer1_ = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2_ = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=0):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        y = self.conv3(y)
        y = self.layer1_(y)
        y = self.layer2_(y)
        y = self.layer3_(y)
        y = self.layer4_(y)
        out = x + y
        out = F.avg_pool2d(out, 4)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


# class resnet_NSCT(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=Categories):
#         super(resnet_NSCT, self).__init__()
#         self.in_planes = 64
#         self.conv_s1 = conv3x3(4, 16)
#         self.conv_l2 = conv3x3(1, 4)
#         self.conv_s2 = conv3x3(4, 16)

#         self.BN = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = conv3x3(4, 64)
#         self.conv2 = conv3x3(64, 64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, ms, pan):
#         l1_ms, s1_ms = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
#         l1_pan, s1_pan = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])

#         l2_ms, s2_ms = CT.contourlet_decompose(l1_ms)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
#         l2_pan, s2_pan = CT.contourlet_decompose(l1_pan)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])

#         l2_pan = self.conv_l2(l2_pan)  # torch.Size([20, 4, 16, 16])
#         l2 = l2_ms + F.interpolate(l2_pan, size=(4, 4))  # torch.Size([20, 4, 4, 4])
#         s2_pan = self.conv_s1(s2_pan)  # torch.Size([20, 16, 16, 16])
#         s2 = s2_ms + F.interpolate(s2_pan, size=(4, 4))  # torch.Size([20, 16, 4, 4])
#         l1_f = CT.contourlet_recompose(l2, s2)  # torch.Size([20, 4, 8, 8])

#         s1_pan = self.conv_s1(s1_pan)  # torch.Size([20, 16, 32, 32])
#         s1 = s1_ms + F.interpolate(s1_pan, size=(8, 8))  # torch.Size([20, 16, 8, 8])
#         f = CT.contourlet_recompose(l1_f, s1)  # # torch.Size([20, 4, 16, 16])

#         # out = self.relu(self.BN(self.conv1(f)))  # torch.Size([20, 64, 32, 32])
#         out = self.relu(self.BN(self.conv1(f)))  # torch.Size([20, 64, 16, 16])
#         out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
#         out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
#         out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
#         out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
#         out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
#         out = out.view(out.size(0), -1)  # torch.Size([20, 512])
#         out = self.linear(out)  # torch.Size([20, 12])
#         return out


# class resnet_NSCT_pan16(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=Categories):
#         super(resnet_NSCT_pan16, self).__init__()
#         self.in_planes = 64
#         self.conv_s1 = conv3x3(4, 16)
#         self.conv_l2 = conv3x3(1, 4)
#         self.conv_s2 = conv3x3(4, 16)

#         self.BN = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = conv3x3(4, 64)
#         self.conv2 = conv3x3(64, 64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, ms, pan):
#         pan = F.interpolate(pan, size=(16, 16))
#         l1_ms, s1_ms = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
#         l1_pan, s1_pan = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])

#         l2_ms, s2_ms = CT.contourlet_decompose(l1_ms)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
#         l2_pan, s2_pan = CT.contourlet_decompose(l1_pan)  # torch.Size([20, 1, 4, 4]) torch.Size([20, 4, 4, 4])

#         l2_pan = self.conv_l2(l2_pan)  # torch.Size([20, 4, 4, 4])
#         s2_pan = self.conv_s1(s2_pan)  # torch.Size([20, 16, 4, 4])

#         l2 = l2_ms + l2_pan  # torch.Size([20, 4, 4, 4])
#         s2 = s2_ms + s2_pan  # torch.Size([20, 16, 4, 4])
#         l1_f = CT.contourlet_recompose(l2, s2)  # torch.Size([20, 4, 8, 8])

#         s1_pan = self.conv_s1(s1_pan)  # torch.Size([20, 16, 8, 8])
#         s1 = s1_ms + s1_pan  # torch.Size([20, 16, 8, 8])
#         f = CT.contourlet_recompose(l1_f, s1)  # # torch.Size([20, 4, 16, 16])

#         # out = self.relu(self.BN(self.conv1(f)))  # torch.Size([20, 64, 32, 32])
#         out = self.relu(self.BN(self.conv1(f)))  # torch.Size([20, 64, 16, 16])
#         out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
#         out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
#         out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
#         out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
#         out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
#         out = out.view(out.size(0), -1)  # torch.Size([20, 512])
#         out = self.linear(out)  # torch.Size([20, 12])
#         return out


def visualize_channels(tensor, num_channels=8, cols=4, name=''):
    """
    可视化指定数量的通道。
    :param tensor: BCHW 形状的张量。
    :param num_channels: 要展示的通道数量。
    :param cols: 每行显示的图像数量。
    """
    import matplotlib.pyplot as plt
    tensor = tensor[0]  # 选择批次中的第一个样本
    rows = num_channels // cols + int(num_channels % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(tensor[i].cpu().detach().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i + 1}-'+name)

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def resnet18():
    return Resnet(BasicBlk, [2, 2, 2, 2])

def Net(args):
    return Resnet(BasicBlk, [2, 2, 2, 2], args['Categories_Number'])


def resnet10():
    return Resnet(BasicBlk, [1, 1, 1, 1])


def resnet10_B():
    return Resnet_Base(BasicBlk, [1, 1, 1, 1])

def resnet18_B():
    return Resnet_Base(BasicBlk, [2, 2, 2, 2])

def resnet10_M():
    return Resnet_M(BasicBlk, [1, 1, 1, 1])

def resnet18_M():
    return Resnet_M(BasicBlk, [2, 2, 2, 2])

def resnet34():
    return Resnet(BasicBlk, [3, 4, 6, 3])


# def resnet10_NSCT():
#     return resnet_NSCT(BasicBlk, [1, 1, 1, 1])


# def resnet18_NSCT():
#     return resnet_NSCT(BasicBlk, [2, 2, 2, 2])


# def resnet18_NSCT_pan16():
#     return resnet_NSCT(BasicBlk, [2, 2, 2, 2])


# def resnet34_NSCT():
#     return resnet_NSCT(BasicBlk, [3, 4, 6, 3])


def resnet50():
    return Resnet(BottleNck, [3, 4, 6, 3])


def resnet101():
    return Resnet(BottleNck, [3, 4, 23, 3])


def resnet152():
    return Resnet(BottleNck, [3, 4, 23, 3])


def test():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8,
    }
    net = Net(cfg).to(device)
    y = net(ms, pan)
    print(y.shape)


def test1():
    # 输入数据 x 的向量维数 10, 设定 LSTM 隐藏层的特征维度 20, 此 model 用 2 个 LSTM 层
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)  # input(seq_len, batch, input_size)
    h0 = torch.randn(2, 3, 20)  # h_0(num_layers * num_directions, batch, hidden_size)
    c0 = torch.randn(2, 3, 20)  # c_0(num_layers * num_directions, batch, hidden_size)
    # output(seq_len, batch, hidden_size * num_directions)
    # h_n(num_layers * num_directions, batch, hidden_size)
    # c_n(num_layers * num_directions, batch, hidden_size)
    output, (hn, cn) = rnn(input, (h0, c0))

    # torch.Size([5, 3, 20]) torch.Size([2, 3, 20]) torch.Size([2, 3, 20])
    print(output.size(), hn.size(), cn.size())


def test2():
    l = [Matrix([3, 2, -1]), Matrix([1, 3, 2]), Matrix([4, 1, 0])]
    # 返回单位化结果
    o2 = GramSchmidt(l, orthonormal=True)  # 注意：orthonormal设为True，执行单位化操作
    print(o2)
    m = np.array(o2)
    # 内积计算，验证施密特正交化结果
    print('任意两向量乘积为：', (m[0] * m[1]).sum())
    print('任一向量的模为：', (m[1] * m[1]).sum())


if __name__ == '__main__':
    test()
