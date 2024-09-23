from PIL import Image
from model.pyct import ContourDec
from model.INN import RevNetModel

from model.resnet import BasicBlk
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.LSTM import ChannelAttention, SpatialAttention
import numpy as np
import time
from model.contourlet_torch import ContourDec
from model.INN import RevNetModel
from model.gmfnet import DH_CGRU_pro


class GMFnetpan(nn.Module):  # 19 (1, 1) 20 (3, 3)
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnetpan, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.device = args['device']
        self.nlevs = 2
        self.address = save_address
        self.convert = nn.Conv2d(1, 4, 1, 1)
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2, cat=1)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2, cat=1)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2, cat=1)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2, cat=1)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2, cat=1)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2, cat=1)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(1, 1))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(1, 1))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        for i in range(3):
            self.inn.append(RevNetModel(num_channels=max_channel * 2 ** i + 4, kernel=3, num_layers=2))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = nn.ModuleList()
        for i in range(3):
            self.SA_p.append(SpatialAttention())
        self.SA_m = nn.ModuleList()
        for i in range(3):
            self.SA_m.append(SpatialAttention())
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

        # self.ct_m = ContourDec(4, self.nlevs, self.device)
        # self.ct_p = ContourDec(1, self.nlevs, self.device)
        self.ct_m = ContourDec(4)
        self.ct_p = ContourDec(1)

    def _make_layer(self, block, planes, num_blocks, stride, cat=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if cat == 1 and stride == strides[0]:
                layers.append(block(self.in_planes + 4, planes, stride))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.convert(pan)

        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)

        # RNN module
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p[2](out_l[2]))
        p_out = torch.concat((p_out, out_l[2]), dim=1)  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m[2](out_s[2]))
        m_out = torch.concat((m_out, out_s[2]), dim=1)
        out = self.inn[0](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p[1](out_l[1]))
        p_out = torch.concat((p_out, out_l[1]), dim=1)
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m[1](out_s[1]))
        m_out = torch.concat((m_out, out_s[1]), dim=1)
        out = self.inn[1](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p[0](out_l[0]))
        p_out = torch.concat((p_out, out_l[0]), dim=1)
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m[0](out_s[0]))
        m_out = torch.concat((m_out, out_s[0]), dim=1)
        out = self.inn[2](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out



def Net(args):
    return GMFnetpan(BasicBlk, [1, 1, 2, 2], args)


def test_net():
    device = 'cuda:0'
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8,
        'device': 'cuda:0'
    }
    net = Net(cfg).to(device)
    y = net(ms, pan)
    print(y.shape)
    # y = net(ms, pan)
    # y = net(ms, pan)
    # y = net(ms, pan)
    # y = net(ms, pan)
    # y = net(ms, pan)
    # y = net(ms, pan)


if __name__ == '__main__':
    test_net()