from model.resnet import BasicBlk
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.LSTM import ChannelAttention, SpatialAttention
import numpy as np
import time
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
from torchvision import transforms
from torchvision.transforms.functional import to_grayscale

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GMFnet(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)
        # print(pan_l1.shape, pan_l2.shape, pan_l3.shape)
        # print(pan_s1.shape, pan_s2.shape, pan_s3.shape)
        # save_image(torch.sigmoid(pan_l1[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l1.png", normalize=True)
        # save_image(torch.sigmoid(pan_l2[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l2.png", normalize=True)
        # save_image(torch.sigmoid(pan_l3[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.log(pan_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s1{i}.png", normalize=True)
        #     save_image(torch.log(pan_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s2{i}.png", normalize=True)
        #     save_image(torch.log(pan_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s3{i}.png", normalize=True)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)
        # print(ms_l1.shape, ms_l2.shape, ms_l3.shape)
        # print(ms_s1.shape, ms_s2.shape, ms_s3.shape)
        # save_image(torch.sigmoid(ms_l1[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l1.png", normalize=True)
        # save_image(torch.sigmoid(ms_l2[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l2.png", normalize=True)
        # save_image(torch.sigmoid(ms_l3[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.relu(ms_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s1{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s2{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s3{i}.png", normalize=True)

        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        # 00, 10, 11, 01

        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        # for i in range(len(out_l)):
        #     save_image(torch.sigmoid(out_l[i]), self.args['RESULT_output'] + f"out_l{i}.png", normalize=True)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]

        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # for i in range(len(out_s)):
        #     save_image(torch.sigmoid(out_s[i][:, :4, :, :]), self.args['RESULT_output'] + f"out_s{i}.png", normalize=True)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        # save_image(pan, self.args['RESULT_output'] + f"pan.png", normalize=True)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)

        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)
        # save_image(m_out[:, :4, :, :], self.args['RESULT_output'] + f"m_out.png", normalize=True)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        for block in self.inn:
            out = block(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        # save_image(out1[:, :4, :, :], self.args['RESULT_output'] + f"out1.png", normalize=True)
        # save_image(out2[:, :4, :, :], self.args['RESULT_output'] + f"out2.png", normalize=True)
        out = F.avg_pool2d(out1 + out2, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class SACA(nn.Module):
    def __init__(self, inplanes, name=''):
        super(SACA, self).__init__()
        self.name = name
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(inplanes)

    def forward(self, tensor):
        if self.name == 'CA':
            out = torch.mul(tensor, self.CA(tensor))
        elif self.name == 'SA':
            out = torch.mul(tensor, self.SA(tensor))
        else:
            out = torch.mul(tensor, self.CA(tensor))
            out = torch.mul(out, self.SA(tensor))
        return out


class ConvGRUCell(nn.Module):
    def __init__(self, in_dim, h_dim, k_size):
        super(ConvGRUCell, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.padding = k_size[0] // 2
        self.bias = False

        self.z = nn.Conv2d(in_channels=in_dim + h_dim, out_channels=h_dim * 2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.r = nn.Conv2d(in_channels=in_dim + h_dim, out_channels=h_dim * 2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.h = nn.Conv2d(in_channels=in_dim + h_dim, out_channels=h_dim * 2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.conv_out = nn.Conv2d(in_channels=h_dim, out_channels=in_dim,
                                  kernel_size=k_size, padding=self.padding,
                                  bias=self.bias)
        self.BN = nn.BatchNorm2d(in_dim)

    def forward(self, in_tensor, cur_state):
        z_com = torch.cat([in_tensor, cur_state], dim=1)
        z_out = self.z(z_com)
        xz, hz = torch.split(z_out, [self.h_dim, self.h_dim], dim=1)
        z_t = torch.sigmoid(xz + hz)

        r_com = torch.cat([in_tensor, cur_state], dim=1)
        r_out = self.r(r_com)
        xr, hr = torch.split(r_out, [self.h_dim, self.h_dim], dim=1)
        r_t = torch.sigmoid(xr + hr)

        h_com = torch.cat([in_tensor, torch.mul(r_t, cur_state)], dim=1)
        h_out = self.h(h_com)
        xh, hh = torch.split(h_out, [self.h_dim, self.h_dim], dim=1)
        h_hat_t = torch.tanh(xh + hh)
        h_t = torch.mul((1 - z_t), cur_state) + torch.mul(z_t, h_hat_t)
        out = torch.sigmoid(self.BN(self.conv_out(h_t)))
        return out, h_t


class H_ConvGRU(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, name=''):
        super(H_ConvGRU, self).__init__()
        self.cell1 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell2 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell3 = ConvGRUCell(in_dim, h_dim, k_size)
        # self.cell4 = ConvGRUCell(in_dim, h_dim, k_size)
        self.SACA = SACA(h_dim, name)

    def init_hidden(self, tensor):
        batch_size = tensor.shape[0]
        height, width = tensor.shape[2], tensor.shape[3]
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, in_tensor, pre_state, state, mode):
        if mode == '1':
            out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state))
            out, cur_state2 = self.cell2(out + in_tensor, self.SACA(cur_state1))
            out, cur_state3 = self.cell3(out + in_tensor, self.SACA(cur_state2))
            # out, cur_state4 = self.cell4(out + in_tensor, self.SACA(cur_state3))
        else:
            if state[0].shape[3] != in_tensor.shape[3]:
                sample = nn.AdaptiveAvgPool2d((in_tensor.shape[2], in_tensor.shape[3]))
                out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state) + sample(state[0]))
                out, cur_state2 = self.cell2(out + in_tensor, self.SACA(cur_state1) + sample(state[1]))
                out, cur_state3 = self.cell3(out + in_tensor, self.SACA(cur_state2) + sample(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(self.upsample(state[3])))
            else:
                out, cur_state1 = self.cell1(in_tensor, pre_state + self.SACA(state[0]))
                out, cur_state2 = self.cell2(out + in_tensor, cur_state1 + self.SACA(state[1]))
                out, cur_state3 = self.cell3(out + in_tensor, cur_state2 + self.SACA(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(state[3]))
        return out, (cur_state1, cur_state2, cur_state3)


class DH_CGRU(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(DH_CGRU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell1 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell2 = ConvGRUCell(in_dim, h_dim, k_size + (2, 2))
        self.cell3 = ConvGRUCell(in_dim, h_dim, k_size + (4, 4))
        self.h_cell1 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell2 = H_ConvGRU(in_dim, h_dim, k_size + (2, 2))
        self.h_cell3 = H_ConvGRU(in_dim, h_dim, k_size + (4, 4))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, in_tensor1, in_tensor2):
        scale1 = in_tensor2  # (20, 1, 8, 8)
        out1, cur_state1 = self.cell1(in_tensor1[2], scale1)
        out_h1, h_state1 = self.h_cell1(out1, cur_state1, cur_state1, mode='1')

        scale2 = self.upsample(scale1)  # (20, 1, 16, 16)
        out2, cur_state2 = self.cell2(in_tensor1[1], scale2)
        out_h2, h_state2 = self.h_cell2(out2, cur_state2, h_state1, mode='2')

        scale3 = self.upsample(scale2)  # (20, 1, 32, 32)
        out3, cur_state3 = self.cell3(in_tensor1[0], scale3)
        out_h3, h_state3 = self.h_cell3(out3, cur_state3, h_state2, mode='2')

        return (out_h1, out_h2, out_h3), (h_state1, h_state2, h_state3)


class DH_CGRU_pro(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(DH_CGRU_pro, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell1 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.cell2 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.cell3 = ConvGRUCell(in_dim, h_dim, (1, 1))
        self.h_cell1 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell2 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell3 = H_ConvGRU(in_dim, h_dim, k_size)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, in_tensor1, in_tensor2):
        out1, cur_state1 = self.cell1(in_tensor1[2], in_tensor2[2])
        out_h1, h_state1 = self.h_cell1(out1, cur_state1, cur_state1, mode='1')

        out2, cur_state2 = self.cell2(in_tensor1[1], in_tensor2[1])
        out_h2, h_state2 = self.h_cell2(out2, cur_state2, h_state1, mode='2')

        out3, cur_state3 = self.cell3(in_tensor1[0], in_tensor2[0])
        out_h3, h_state3 = self.h_cell3(out3, cur_state3, h_state2, mode='2')

        return (out_h1, out_h2, out_h3), (h_state1, h_state2, h_state3)


class GMFnet_baseline(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_baseline, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_inn(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_inn, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        # pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        # pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        # pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)
        # print(pan_l1.shape, pan_l2.shape, pan_l3.shape)
        # print(pan_s1.shape, pan_s2.shape, pan_s3.shape)
        # save_image(torch.sigmoid(pan_l1[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l1.png", normalize=True)
        # save_image(torch.sigmoid(pan_l2[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l2.png", normalize=True)
        # save_image(torch.sigmoid(pan_l3[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.log(pan_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s1{i}.png", normalize=True)
        #     save_image(torch.log(pan_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s2{i}.png", normalize=True)
        #     save_image(torch.log(pan_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s3{i}.png", normalize=True)

        # ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        # ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        # ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)
        # print(ms_l1.shape, ms_l2.shape, ms_l3.shape)
        # print(ms_s1.shape, ms_s2.shape, ms_s3.shape)
        # save_image(torch.sigmoid(ms_l1[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l1.png", normalize=True)
        # save_image(torch.sigmoid(ms_l2[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l2.png", normalize=True)
        # save_image(torch.sigmoid(ms_l3[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.relu(ms_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s1{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s2{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s3{i}.png", normalize=True)

        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        # 00, 10, 11, 01

        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        # for i in range(len(out_l)):
        #     save_image(torch.sigmoid(out_l[i]), self.args['RESULT_output'] + f"out_l{i}.png", normalize=True)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]

        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # for i in range(len(out_s)):
        #     save_image(torch.sigmoid(out_s[i][:, :4, :, :]), self.args['RESULT_output'] + f"out_s{i}.png", normalize=True)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        # save_image(pan, self.args['RESULT_output'] + f"pan.png", normalize=True)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        # p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        # p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        # p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)

        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        # m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        # m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        # m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)
        # save_image(m_out[:, :4, :, :], self.args['RESULT_output'] + f"m_out.png", normalize=True)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        for block in self.inn:
            out = block(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        # save_image(out1[:, :4, :, :], self.args['RESULT_output'] + f"out1.png", normalize=True)
        # save_image(out2[:, :4, :, :], self.args['RESULT_output'] + f"out2.png", normalize=True)
        out = F.avg_pool2d(out1 + out2, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_L(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_L, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)
        # print(pan_l1.shape, pan_l2.shape, pan_l3.shape)
        # print(pan_s1.shape, pan_s2.shape, pan_s3.shape)
        # save_image(torch.sigmoid(pan_l1[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l1.png", normalize=True)
        # save_image(torch.sigmoid(pan_l2[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l2.png", normalize=True)
        # save_image(torch.sigmoid(pan_l3[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.log(pan_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s1{i}.png", normalize=True)
        #     save_image(torch.log(pan_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s2{i}.png", normalize=True)
        #     save_image(torch.log(pan_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s3{i}.png", normalize=True)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)
        # print(ms_l1.shape, ms_l2.shape, ms_l3.shape)
        # print(ms_s1.shape, ms_s2.shape, ms_s3.shape)
        # save_image(torch.sigmoid(ms_l1[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l1.png", normalize=True)
        # save_image(torch.sigmoid(ms_l2[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l2.png", normalize=True)
        # save_image(torch.sigmoid(ms_l3[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.relu(ms_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s1{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s2{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s3{i}.png", normalize=True)

        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        # 00, 10, 11, 01

        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        # for i in range(len(out_l)):
        #     save_image(torch.sigmoid(out_l[i]), self.args['RESULT_output'] + f"out_l{i}.png", normalize=True)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]

        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # for i in range(len(out_s)):
        #     save_image(torch.sigmoid(out_s[i][:, :4, :, :]), self.args['RESULT_output'] + f"out_s{i}.png", normalize=True)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        # save_image(pan, self.args['RESULT_output'] + f"pan.png", normalize=True)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)

        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        # m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        # m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        # m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)
        # save_image(m_out[:, :4, :, :], self.args['RESULT_output'] + f"m_out.png", normalize=True)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        # save_image(out1[:, :4, :, :], self.args['RESULT_output'] + f"out1.png", normalize=True)
        # save_image(out2[:, :4, :, :], self.args['RESULT_output'] + f"out2.png", normalize=True)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_B(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_B, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)
        # print(pan_l1.shape, pan_l2.shape, pan_l3.shape)
        # print(pan_s1.shape, pan_s2.shape, pan_s3.shape)
        # save_image(torch.sigmoid(pan_l1[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l1.png", normalize=True)
        # save_image(torch.sigmoid(pan_l2[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l2.png", normalize=True)
        # save_image(torch.sigmoid(pan_l3[:, :4, :, :]), self.args['RESULT_output'] + f"pan_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.log(pan_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s1{i}.png", normalize=True)
        #     save_image(torch.log(pan_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s2{i}.png", normalize=True)
        #     save_image(torch.log(pan_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"pan_s3{i}.png", normalize=True)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)
        # print(ms_l1.shape, ms_l2.shape, ms_l3.shape)
        # print(ms_s1.shape, ms_s2.shape, ms_s3.shape)
        # save_image(torch.sigmoid(ms_l1[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l1.png", normalize=True)
        # save_image(torch.sigmoid(ms_l2[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l2.png", normalize=True)
        # save_image(torch.sigmoid(ms_l3[:, :4, :, :]), self.args['RESULT_output'] + f"ms_l3.png", normalize=True)
        # for i in range(4):
        #     save_image(torch.relu(ms_s1[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s1{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s2[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s2{i}.png", normalize=True)
        #     save_image(torch.relu(ms_s3[:, i:i+1, :, :]), self.args['RESULT_output'] + f"ms_s3{i}.png", normalize=True)

        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        # 00, 10, 11, 01

        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        # for i in range(len(out_l)):
        #     save_image(torch.sigmoid(out_l[i]), self.args['RESULT_output'] + f"out_l{i}.png", normalize=True)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]

        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # for i in range(len(out_s)):
        #     save_image(torch.sigmoid(out_s[i][:, :4, :, :]), self.args['RESULT_output'] + f"out_s{i}.png", normalize=True)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        # save_image(pan, self.args['RESULT_output'] + f"pan.png", normalize=True)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        # p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        # p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        # p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)

        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)
        # save_image(m_out[:, :4, :, :], self.args['RESULT_output'] + f"m_out.png", normalize=True)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        # save_image(out1[:, :4, :, :], self.args['RESULT_output'] + f"out1.png", normalize=True)
        # save_image(out2[:, :4, :, :], self.args['RESULT_output'] + f"out2.png", normalize=True)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_BL(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_BL, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(1, 1))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(1, 1))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)

        # 00, 10, 11, 01
        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))

        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])

        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        # out = p_out + m_out
        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_concat(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_concat, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
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

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel_size=3, num_layers=6))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

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
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)

        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.concat((p_out, out_l[2]), dim=1)  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.concat((p_out, out_l[1]), dim=1)
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.concat((p_out, out_l[0]), dim=1)
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.concat((m_out, out_s[2]), dim=1)
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.concat((m_out, out_s[1]), dim=1)
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.concat((m_out, out_s[0]), dim=1)
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        for block in self.inn:
            out = block(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(out1 + out2, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_new(nn.Module):  # 14 (3,3) 18(1, 1)
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_new, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel * 2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel * 4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(1, 1))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(1, 1))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        for i in range(3):
            self.inn.append(RevNetModel(num_channels=max_channel * 2 ** i, kernel_size=3, num_layers=2))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
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
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        out = self.inn[0](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        out = self.inn[1](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        out = self.inn[2](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_new_cat(nn.Module):  # 17
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_new_cat, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
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

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(3, 3))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(3, 3))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        for i in range(3):
            self.inn.append(RevNetModel(num_channels=max_channel * 2 ** i + 4, kernel_size=3, num_layers=2))
        # for i in range(len(inn_list)):
        #     self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=inn_list[i], num_layers=10))
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

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
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
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
        p_out = torch.concat((p_out, out_l[2]), dim=1)  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.concat((m_out, out_s[2]), dim=1)
        out = self.inn[0](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.concat((p_out, out_l[1]), dim=1)
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.concat((m_out, out_s[1]), dim=1)
        out = self.inn[1](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.concat((p_out, out_l[0]), dim=1)
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.concat((m_out, out_s[0]), dim=1)
        out = self.inn[2](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)

        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class GMFnet_new_cat1jia(nn.Module):  # 19 (1, 1) 20 (3, 3)
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(GMFnet_new_cat1jia, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.device = args['device']
        self.nlevs = 2
        self.address = save_address
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
        ms = self.upsample(m)
        # ms = F.interpolate(m, size=[64, 64])
        # xianhua(ms, 'm.jpg')
        # xianhua(pan, 'p.jpg')
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        t1 = time.time()
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)

        ms_l1, ms_s1 = ContourDec(self.nlevs)(ms)
        ms_l2, ms_s2 = ContourDec(self.nlevs)(ms_l1)
        ms_l3, ms_s3 = ContourDec(self.nlevs)(ms_l2)

        # visualize_channels(pan_l1, 4, 4, 'pl1')
        # visualize_channels(pan_s1, 4, 4, 'ps1')

        t2 = time.time()
        # visualize_channels(ms_l1, 4, 4, 'ml1')
        # visualize_channels(ms_s1, 16, 4, 'ms1')

        # RNN module     fddd
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # visualize_channels(out_l[2], 4, 2, "out_l2")
        # visualize_channels(out_s[2], 4, 2, "out_s2")
        # visualize_channels(out_l[1], 4, 2, "out_l1")
        # visualize_channels(out_s[1], 4, 2, "out_s1")
        # visualize_channels(out_l[0], 4, 2, "out_l0")

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        # visualize_channels(m, 8, 4, 'm')
        # visualize_channels(p, 8, 4, 'p')
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        # visualize_channels(p_out, 8, 4, 'p_1')
        p_out = torch.mul(p_out, self.SA_p[2](out_l[2]))
        p_out = torch.concat((p_out, out_l[2]), dim=1)  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        # visualize_channels(m_out, 8, 4, 'm_1')
        m_out = torch.mul(m_out, self.SA_m[2](out_s[2]))
        m_out = torch.concat((m_out, out_s[2]), dim=1)
        out = self.inn[0](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)
        # visualize_channels(p_out, 8, 4, 'p_out1')
        # visualize_channels(m_out, 8, 4, 'm_out1')

        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        # visualize_channels(p_out, 8, 4, 'p_2')
        p_out = torch.mul(p_out, self.SA_p[1](out_l[1]))
        p_out = torch.concat((p_out, out_l[1]), dim=1)
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        # visualize_channels(m_out, 8, 4, 'm_2')
        m_out = torch.mul(m_out, self.SA_m[1](out_s[1]))
        m_out = torch.concat((m_out, out_s[1]), dim=1)
        out = self.inn[1](torch.cat((p_out, m_out), dim=1))
        p_out, m_out = torch.chunk(out, 2, dim=1)
        # visualize_channels(p_out, 8, 4, 'p_out2')
        # visualize_channels(m_out, 8, 4, 'm_out2')

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

        # for block in self.inn:
        #     out = block(torch.cat((p_out, m_out), dim=1))
        # out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(m_out + p_out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        time3 = time.time()
        # print(t2 - t1, time3 - t2)
        return out


def Net(args):
    return GMFnet_new_cat1jia(BasicBlk, [1, 1, 2, 2], args)


# all 10 baseline 11 B 12 L 13 BL 14 inn 15
# baseline 0 B 1 L 2 BL 3 inn 4 ALL 5 #
# BL|k=1 6 [2, 2, 2, 2] 7


def img_channels(img, num_channel=8, cols=4, name=''):
    from PIL import Image
    img = img[0].cpu().detach().numpy()

    channels = img.shape[0]
    print(img.shape)
    if channels == 1:
        img = np.concatenate([img, img, img, img], axis=0)
        img = np.transpose(img, (1, 2, 0))
    else:
        # img = img[:3, :, :]
        img = np.transpose(img, (1, 2, 0))

    save_img = np.uint8(img * 255).astype('uint8')
    save_img = Image.fromarray(save_img, "CMYK")
    save_img.save(name+'.jpg')


def visualize_channels(tensor, num_channels=8, cols=4, name=''):

    """
    可视化指定数量的通道。
    :param tensor: BCHW 形状的张量。
    :param num_channels: 要展示的通道数量。
    :param cols: 每行显示的图像数量。
    """
    import matplotlib.pyplot as plt
    tensor = tensor[0]  # 选择批次中的第一个样本
    channels = tensor.shape[0]  # 获取通道数

    # 如果通道数为1，仅展示这一个通道
    if channels == 1:
        plt.imshow(tensor[0].cpu().detach().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title('Single Channel_'+name)
        plt.savefig('Single Channel_'+name)
        plt.show()

        return

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
    plt.savefig(f'Channel {i + 1}-'+name)
    plt.show()


def equalize_histogram(band):
    hist, bins = np.histogram(band.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[band]


def xianhua(img, name, equalize=1):
    img = img[0].cpu().detach().numpy()

    if img.shape[0] == 4:
        band_data = img[(2, 1, 0), :, :]
        scaled_data = []
        for i, band in enumerate(band_data):
            band_min, band_max = band.min(), band.max()
            scaled_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            if equalize:
                scaled_band = equalize_histogram(scaled_band)
            # scaled_band = adjust_contrast(scaled_band, 0.95)
            # scaled_band = adjust_exposure(scaled_band, 0.95)
            scaled_data.append(scaled_band)

        processed_array = np.dstack(scaled_data)
        # processed_array = adjust_brightness_hsv(processed_array, 0.85)
    elif img.shape[0] == 1:
        band = img[0]
        band_min, band_max = band.min(), band.max()
        processed_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
        if equalize:
            processed_band = equalize_histogram(processed_band)
        processed_array = processed_band

    else:
        raise ValueError("Unsupported image type. Please use 'multispectral' or 'pan'.")

    result = Image.fromarray(processed_array, 'RGB' if img.shape[0] == 4 else 'L')
    result.show()
    result.save(name)


def test_net():
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


