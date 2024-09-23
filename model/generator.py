import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)  if use_act else nn.Identity(),
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, features=64, residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features, 1, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock(features, features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(features*2, features*4, kernel_size=3, stride=2, padding=1),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(features*4) for _ in range(residuals)]
        )

        self.up_blocks = nn.Sequential(
            ConvBlock(features*4, features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(features*2, features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.last = nn.Conv2d(features, img_channels, 7, 1, 3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)  # torch.Size([5, 64, 16, 16])
        x = self.down_blocks(x)  # torch.Size([5, 256, 4, 4])
        x = self.res_blocks(x)  # torch.Size([5, 256, 4, 4])
        x = self.up_blocks(x)  # torch.Size([5, 64, 16, 16])
        x = self.last(x)
        return torch.tanh(x)


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


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


class Resnet(nn.Module):
    def __init__(self, block, num_blocks):
        super(Resnet, self).__init__()
        num_channels = 4
        in_chan = 64
        self.in_planes = in_chan
        self.conv_in = conv3x3(4, in_chan)
        self.BN = nn.BatchNorm2d(in_chan)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, in_chan, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, in_chan, num_blocks[3], stride=1)

        self.conv_out = conv3x3(in_chan, num_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.BN(self.conv_in(x)))
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = torch.sigmoid(self.conv_out(out))
        return out


def Net():
    # return Generator(img_channels=4)
    return Resnet(BasicBlk, [2, 2, 2, 2])

def test():
    x = torch.randn((5, 4, 16, 16))
    gen = Net()#Generator(img_channels=4)
    preds = gen(x)
    print(preds.shape)
    # print(gen)


if __name__ == "__main__":
    test()