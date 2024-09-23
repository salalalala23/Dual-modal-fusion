import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as config
Categories = config.Categories
device = config.DEVICE

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=Categories):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

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

    def forward(self, x, y=0, mode='2'):
        if mode == '2':
            out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
            out = F.relu(self.bn2(self.conv2(out)))  # torch.Size([20, 64, 16, 16])
        elif mode == '1':
            out = F.relu(self.bn1(self.conv1(x)))  # torch.Size([20, 64, 16, 16])
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


# class ResNet4(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=Categories):
#         super(ResNet4, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x, y, z, q):
#         # out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
#         out = F.relu(self.bn1(self.conv1(x)))  # torch.Size([20, 64, 16, 16])
#         out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
#         out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
#         out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
#         out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
#         out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
#         out = out.view(out.size(0), -1)  # torch.Size([20, 512])
#         out = self.linear(out)  # torch.Size([20, 12])
#         return out


# class ResNet_dual(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=Categories):
#         super(ResNet_dual, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#
#         self.layer5 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer6 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer7 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x, y):
#         # out = torch.concat([x, F.interpolate(y, size=(16, 16))], dim=1)
#         out1 = F.relu(self.bn1(self.conv1(x)))  # torch.Size([20, 64, 16, 16])
#         out1 = self.layer1(out1)  # torch.Size([20, 64, 16, 16])
#         out1 = self.layer2(out1)  # torch.Size([20, 128, 8, 8])
#         out1 = self.layer3(out1)  # torch.Size([20, 256, 4, 4])
#         out1 = self.layer4(out1)  # torch.Size([20, 512, 2, 2])
#
#         out2 = F.relu(self.bn1(self.conv1(y)))
#         out2 = self.layer1(out2)
#         out2 = self.layer5(out2)
#         out2 = self.layer6(out2)
#         out2 = self.layer7(out2)
#
#         out = torch.cat([out1, out2])
#         out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
#         out = out.view(out.size(0), -1)  # torch.Size([20, 512])
#         out = self.linear(out)  # torch.Size([20, 12])
#         return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
#
#
# def ResNet18_2():
#     return ResNet2(BasicBlock, [2, 2, 2, 2])
#
# def ResNet18_4():
#     return ResNet4(BasicBlock, [2, 2, 2, 2])

#
# def ResNet18_dual():
#     return ResNet_dual(BasicBlock, [2, 2, 2, 2])


if __name__ == '__main__':
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    net = ResNet18().to(device)
    y = net(ms, pan)
    print(y.shape)
