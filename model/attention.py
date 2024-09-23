import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=inplanes // ratio, out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)  # torch.size([batch_size, 1, width, height])
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):  # 这里默认为16,如果特征图通道数较小，可以适当调整
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn((20, 16, 16, 16))
    model = SqueezeExcitation(16)
    y = model(x)
    print(y.shape)
