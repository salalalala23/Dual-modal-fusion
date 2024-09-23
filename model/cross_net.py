import torch
import torch.nn as nn
from torch.nn import functional as F
from model.generator import Generator


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
        out = self.shortcut(x) + out
        out = self.relu(out)
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


# BasicBlock
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out


class Cross_net(nn.Module):
    def __init__(self, args, num_blocks=[2, 2, 2, 2]):
        super(Cross_net, self).__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.in_planes = 256
        self.m_layer1_1 = self._make_layer(BasicBlk, 256, num_blocks[0], stride=2)
        self.m_layer2_1 = self._make_layer(BasicBlk, 256, num_blocks[1], stride=1)
        self.m_layer3_1 = self._make_layer(BasicBlk, 512, num_blocks[2], stride=1)
        self.m_layer4_1 = self._make_layer(BasicBlk, 512, num_blocks[3], stride=1)
        self.in_planes = 256
        self.p_layer1_1 = self._make_layer(BasicBlk, 256, num_blocks[0], stride=2)
        self.p_layer2_1 = self._make_layer(BasicBlk, 256, num_blocks[1], stride=1)
        self.p_layer3_1 = self._make_layer(BasicBlk, 512, num_blocks[2], stride=1)
        self.p_layer4_1 = self._make_layer(BasicBlk, 512, num_blocks[3], stride=1)

        self.in_planes = 64
        self.layer1_2 = self._make_layer(BasicBlk, 128, num_blocks[0], stride=2)
        self.layer2_2 = self._make_layer(BasicBlk, 256, num_blocks[1], stride=2)
        self.layer3_2 = self._make_layer(BasicBlk, 512, num_blocks[2], stride=2)
        self.layer4_2 = self._make_layer(BasicBlk, 512, num_blocks[3], stride=1)
        # self.m_layer2 = self._make_layer(BasictranBlk, 256, num_blocks[1], stride=1)

        self.blk1234_1 = ResBlk(64, 64, stride=1)
        self.blk12_1 = ResBlk(128, 128, stride=1)
        self.blk34_1 = ResBlk(128, 128, stride=1)
        self.blk13_1 = ResBlk(256, 256, stride=1)
        self.blk24_1 = ResBlk(256, 256, stride=1)
        self.sa_m = SpatialAttention()
        self.sa_p = SpatialAttention()

        self.m_linear1 = nn.Linear(512, args['Categories_Number'])
        self.p_linear1 = nn.Linear(512, args['Categories_Number'])
        self.linear2 = nn.Linear(512, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, gen_p, gen_m):
        bs = int(x.shape[0] / 4)
        out = F.gelu(self.conv(x))

        m_out = gen_p.down_blocks(torch.concat([out[:bs], out[2 * bs:3 * bs]]))
        p_out = gen_m.down_blocks(torch.concat([out[bs:2 * bs], out[-bs:]]))
        m, p, gm, gp = m_out[:bs], p_out[:bs], m_out[bs:2 * bs], p_out[bs:2 * bs]
        m = torch.mul(m, self.sa_m(gm))
        p = torch.mul(p, self.sa_p(gp))
        out = torch.concat([m, p, gm, gp])
        m = self.m_layer1_1(m)
        m = self.m_layer2_1(m)
        m = self.m_layer3_1(m)
        m = self.m_layer4_1(m)
        m = F.avg_pool2d(m, 2)
        m = m.view(m.size(0), -1)
        m = self.m_linear1(m)

        p = self.p_layer1_1(p)
        p = self.p_layer2_1(p)
        p = self.p_layer3_1(p)
        p = self.p_layer4_1(p)
        p = F.avg_pool2d(p, 2)
        p = p.view(p.size(0), -1)
        p = self.p_linear1(p)
        l_out = torch.concat([m, p])

        m_out = gen_p.up_blocks(torch.concat([out[:bs], out[2 * bs:3 * bs]]))
        p_out = gen_m.up_blocks(torch.concat([out[bs:2 * bs], out[-bs:]]))

        out = torch.concat([m_out[:bs], p_out[:bs], m_out[bs:], p_out[bs:]])
        out = self.layer1_2(out)
        out = self.layer2_2(out)
        out = self.layer3_2(out)
        out = self.layer4_2(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear2(out)
        return torch.concat([out, l_out])


def Net(args):
    return Cross_net(args)


def test():
    args = {
        'Categories_Number': 8
    }
    m = Generator(img_channels=4)
    p = Generator(img_channels=4)
    net = Net(args)
    y = net(torch.randn(80, 4, 16, 16), m, p)
    # print(y.size())


if __name__ == '__main__':
    test()
    a = torch.randn([10, 8])
    b = a / 2
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 = torch.tensor([1.000, 0.000, 0.000, 0.000, 0.0, 0.000, 0.0, 0.0])
    y2 = torch.tensor([2.0, 1.9, 0.2, 0.6, 0.8, 0.9, 0.4, 0.5])
    y3 = torch.tensor([2.0, 0.1, 1.5, 1.4, 2.1, 0.1, 0.9, 1.5])
    y4 = torch.tensor([2.0, 2.1, 0.2, 0.6, 0.8, 0.9, 0.4, 0.5])
    y5 = torch.tensor([1.95, 1.9, 0.2, 0.6, 0.8, 0.9, 0.4, 0.5])
    kl1 = F.kl_div(y3.softmax(dim=-1).log(), y2.softmax(dim=-1), reduction='batchmean')
    kl2 = F.kl_div(y4.softmax(dim=-1).log(), y2.softmax(dim=-1), reduction='batchmean')
    kl3 = F.kl_div(y5.softmax(dim=-1).log(), y2.softmax(dim=-1), reduction='batchmean')
    print(y1.softmax(dim=-1), y2.softmax(dim=-1))
    print(F.kl_div(y2.softmax(dim=-1).log(), y1.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y3.softmax(dim=-1).log(), y1.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y4.softmax(dim=-1).log(), y1.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y5.softmax(dim=-1).log(), y1.softmax(dim=-1), reduction='batchmean'))
    print(F.kl_div(y1.softmax(dim=-1).log(), y2.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y1.softmax(dim=-1).log(), y3.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y1.softmax(dim=-1).log(), y4.softmax(dim=-1), reduction='batchmean'),
          F.kl_div(y1.softmax(dim=-1).log(), y5.softmax(dim=-1), reduction='batchmean'))
    print(kl1, kl2, kl3)
    import matplotlib.pyplot as plt
    plt.plot(x, y1.softmax(dim=-1), label='Sequence 1')
    plt.plot(x, y2.softmax(dim=-1), label='Sequence 2')
    plt.plot(x, y3.softmax(dim=-1), label='Sequence 3')
    plt.plot(x, y4.softmax(dim=-1), label='Sequence 4')
    plt.plot(x, y5.softmax(dim=-1), label='Sequence 5')
    # 设置图例
    plt.legend()

    # 设置图表标题和轴标签
    plt.title('Sequence Curves')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示网格线
    plt.grid(True)

    # 显示图表
    plt.show()
