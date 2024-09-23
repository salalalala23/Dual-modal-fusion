from model.resnet import BasicBlk
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.LSTM import ChannelAttention, SpatialAttention
import numpy as np
import time
from model.contourlet_torch import ContourDec
import os
from torchvision.utils import save_image
from model.INN import RevNetModel
from torchvision import transforms
from torchvision.transforms.functional import to_grayscale
from model.pycontourlet.pycontourlet4d.pycontourlet import batch_multi_channel_pdfbdec

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of same img dimension together."""
    # Collect tensor with same dimension into a dict of list
    output = {}

    # Input is list
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                for j in range(len(x[i])):
                    shape = tuple(x[i][j].shape)
                    if shape in output.keys():
                        output[shape].append(x[i][j])
                    else:
                        output[shape] = [x[i][j]]
            else:
                shape = tuple(x[i].shape)
                if shape in output.keys():
                    output[shape].append(x[i])
                else:
                    output[shape] = [x[i]]
    else:
        for k in x.keys():
            shape = tuple(x[k].shape[2:4])
            if shape in output.keys():
                output[shape].append(x[k])
            else:
                output[shape] = [x[k]]

    # Concat the list of tensors into single tensor
    for k in output.keys():
        output[k] = torch.cat(output[k], dim=1)

    return output


class CLSTM8(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, inn_list=[3, 5, 7], variant="SSF", spec_type="all", save_address=''):
        super(CLSTM8, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.args = args

        self.nlevs = 2
        self.address = save_address
        self.conv_m = nn.Conv2d(4, max_channel, 1, 1)
        self.conv_p = nn.Conv2d(1, max_channel, 1, 1)

        self.in_planes = max_channel
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)
        self.in_planes = max_channel
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=4, h_dim=1, k_size=(1, 1))
        self.GRU_s = DH_CGRU_pro(in_dim=4, h_dim=16, k_size=(1, 1))
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = nn.ModuleList()
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=7, num_layers=3))
        # self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=5, num_layers=3))
        self.inn.append(RevNetModel(num_channels=max_channel * 8, kernel=3, num_layers=6))
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
        visualize_channels(m, 4, 4, 'ms')
        visualize_channels(pan, 1, 4, 'pan')
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        pan_l1, pan_s1 = ContourDec(self.nlevs)(pan)
        pan_l2, pan_s2 = ContourDec(self.nlevs)(pan_l1)
        pan_l3, pan_s3 = ContourDec(self.nlevs)(pan_l2)
        visualize_channels(pan_s1, 4, 4, 'pan_s1')
        visualize_channels(pan_l1, 1, 4, 'pan_l1')
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
        visualize_channels(ms_s1, 4, 4, 'ms_s1')
        visualize_channels(ms_l1, 4, 4, 'ms_l1')
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
        # out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        out_l, state_l = self.GRU_l((ms_l1, ms_l2, ms_l3), (pan_l1, pan_l2, pan_l3))
        # visualize_channels(out_l[0], 4, 4, 'outl0')
        # visualize_channels(out_l[1], 4, 4, 'outl1')
        visualize_channels(out_l[2], 4, 4, 'outl2')
        # for i in range(len(out_l)):
        #     save_image(torch.sigmoid(out_l[i]), self.args['RESULT_output'] + f"out_l{i}.png", normalize=True)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        # out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), (ms_s1, ms_s2, ms_s3))
        # visualize_channels(out_s[0], 4, 4, 'outs0')
        # visualize_channels(out_s[1], 4, 4, 'outs1')
        visualize_channels(out_s[2], 4, 4, 'outs2')
        # for i in range(len(out_s)):
        #     save_image(torch.sigmoid(out_s[i][:, :4, :, :]), self.args['RESULT_output'] + f"out_s{i}.png", normalize=True)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        # save_image(pan, self.args['RESULT_output'] + f"pan.png", normalize=True)
        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        visualize_channels(p_out, 8, 4, 'pout1')
        visualize_channels(self.SA_p(out_l[2]), 1, 4, 'SA_p')
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        visualize_channels(p_out, 8, 4, 'pout2')
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        # visualize_channels(p_out, 8, 4, 'pout3')
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        # visualize_channels(p_out, 8, 4, 'pout4')
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        # visualize_channels(p_out, 8, 4, 'pout5')
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        # visualize_channels(p_out, 8, 4, 'pout6')
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # visualize_channels(p_out, 8, 4, 'pout')
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])
        # save_image(p_out[:, :4, :, :], self.args['RESULT_output'] + f"p_out.png", normalize=True)


        # save_image(ms, self.args['RESULT_output'] + f"ms.png", normalize=True)
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        visualize_channels(m_out, 8, 4, 'mout1')
        visualize_channels(self.SA_m(out_s[2]), 1, 4, 'SA_m')
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        visualize_channels(m_out, 8, 4, 'mout2')
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        # visualize_channels(m_out, 8, 4, 'mout3')
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        # visualize_channels(m_out, 8, 4, 'mout4')
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        # visualize_channels(m_out, 8, 4, 'mout5')
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        # visualize_channels(m_out, 8, 4, 'mout6')
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # visualize_channels(m_out, 8, 4, 'mout')
        # m_out = F.avg_pool2d(m_out, 4)
        # save_image(m_out[:, :4, :, :], self.args['RESULT_output'] + f"m_out.png", normalize=True)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        for block in self.inn:
            out = block(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        # visualize_channels(out1, 8, 4, 'out1')
        # visualize_channels(out2, 8, 4, 'out2')
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
        self.padding = k_size[0]//2
        self.bias = False

        self.z = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.r = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
                           kernel_size=k_size, padding=self.padding,
                           bias=self.bias)
        self.h = nn.Conv2d(in_channels=in_dim+h_dim, out_channels=h_dim*2,
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
        self.cell4 = ConvGRUCell(in_dim, h_dim, k_size)
        self.SACA = SACA(h_dim, name)

    def init_hidden(self, tensor):
        batch_size = tensor.shape[0]
        height, width = tensor.shape[2], tensor.shape[3]
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, in_tensor, pre_state, state, mode):
        if mode == '1':
            out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state))
            out, cur_state2 = self.cell2(out+in_tensor, self.SACA(cur_state1))
            out, cur_state3 = self.cell3(out+in_tensor, self.SACA(cur_state2))
            # out, cur_state4 = self.cell4(out + in_tensor, self.SACA(cur_state3))
        else:
            if state[0].shape[3] != in_tensor.shape[3]:
                sample = nn.AdaptiveAvgPool2d((in_tensor.shape[2], in_tensor.shape[3]))
                out, cur_state1 = self.cell1(in_tensor, self.SACA(pre_state) + sample(state[0]))
                out, cur_state2 = self.cell2(out+in_tensor, self.SACA(cur_state1)+sample(state[1]))
                out, cur_state3 = self.cell3(out+in_tensor, self.SACA(cur_state2)+sample(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(self.upsample(state[3])))
            else:
                out, cur_state1 = self.cell1(in_tensor, pre_state + self.SACA(state[0]))
                out, cur_state2 = self.cell2(out+in_tensor, cur_state1+self.SACA(state[1]))
                out, cur_state3 = self.cell3(out+in_tensor, cur_state2+self.SACA(state[2]))
                # out, cur_state4 = self.cell4(out + in_tensor, cur_state3 + self.SACA(state[3]))
        return out, (cur_state1, cur_state2, cur_state3)


class DH_CGRU(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(DH_CGRU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell1 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell2 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell3 = ConvGRUCell(in_dim, h_dim, k_size)
        self.h_cell1 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell2 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell3 = H_ConvGRU(in_dim, h_dim, k_size)
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


class DH_CGRU_s(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(DH_CGRU_s, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell1 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell2 = ConvGRUCell(in_dim, h_dim, k_size)
        self.cell3 = ConvGRUCell(in_dim, h_dim, k_size)
        self.h_cell1 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell2 = H_ConvGRU(in_dim, h_dim, k_size)
        self.h_cell3 = H_ConvGRU(in_dim, h_dim, k_size)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, in_tensor1, in_tensor2):
        scale1 = in_tensor1  # (20, 4, 8, 8)
        out1, cur_state1 = self.cell1(scale1, in_tensor2[2])
        out_h1, h_state1 = self.h_cell1(out1, cur_state1, cur_state1, mode='1')

        scale2 = self.upsample(scale1)  # (20, 4, 16, 16)
        out2, cur_state2 = self.cell2(scale2, in_tensor2[1])
        out_h2, h_state2 = self.h_cell2(out2, cur_state2, h_state1, mode='2')

        scale3 = self.upsample(scale2)  # (20, 4, 32, 32)
        out3, cur_state3 = self.cell3(scale3, in_tensor2[0])
        out_h3, h_state3 = self.h_cell3(out3, cur_state3, h_state2, mode='2')

        return (out_h1, out_h2, out_h3), (h_state1, h_state2, h_state3)


class DH_CGRU_mini(nn.Module):
    def __init__(self, in_dim, h_dim, k_size, mode=''):
        super(DH_CGRU_mini, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                              kernel_size=(1, 1), stride=1)
        self.cell = ConvGRUCell(in_dim, h_dim, k_size)
        self.h_cell = H_ConvGRU(in_dim, h_dim, k_size)

    def forward(self, in_tensor1, in_tensor2, state, mode):
        sample = nn.AdaptiveAvgPool2d((in_tensor1.shape[2], in_tensor1.shape[3]))
        in_tensor2 = sample(in_tensor2)
        out, cur_state = self.cell(in_tensor1, in_tensor2)
        out_h, h_state = self.h_cell(out, cur_state, state, mode)
        return out_h, h_state


class CLSTM1(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, variant="SSF", spec_type="all"):
        super(CLSTM1, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.in_planes_1 = 64
        self.in_planes_2 = 64
        self.nlevs = [2]

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer_1(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer_1(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer_1(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer_1(block, max_channel*8, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer_2(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer_2(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer_2(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer_2(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer_2(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU(in_dim=1, h_dim=4, k_size=(1, 1), mode='SA')
        self.GRU_s = DH_CGRU(in_dim=4, h_dim=16, k_size=(1, 1), mode='SA')
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))

        self.SA_p1 = SpatialAttention()
        self.SA_p2 = SpatialAttention()
        self.SA_p3 = SpatialAttention()
        self.SA_m1 = SpatialAttention()
        self.SA_m2 = SpatialAttention()
        self.SA_m3 = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer_1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_1, planes, stride))
            self.in_planes_1 = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_2, planes, stride))
            self.in_planes_2 = planes * block.expansion
        return nn.Sequential(*layers)

    def __pdfbdec(self, x, nlevs=[0, 3, 3, 3], method="resize"):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="9-7", dfilt="vk", nlevs=nlevs, device=device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, ms, pan):
        # ms:[20, 4, 16, 16] pan:[20, 1, 64, 64]
        # CT module
        # pan_coefs, _ = self.__pdfbdec(pan, nlevs=[0, 2, 2, 2])
        # for key in pan_coefs:
        #     print(key, pan_coefs[key].shape)
        # pan_s1, pan_s2, pan_s3 = pan_coefs[0], pan_coefs[1], pan_coefs[2]
        pan_coefs1, _ = self.__pdfbdec(pan, nlevs=self.nlevs)
        pan_l1, pan_s1 = pan_coefs1[0][:, :pan.shape[1], :, :].to(device), \
            pan_coefs1[0][:, pan.shape[1]:, :, :].to(device)
        pan_coefs2, _ = self.__pdfbdec(pan_l1, nlevs=self.nlevs)
        pan_l2, pan_s2 = pan_coefs2[0][:, :pan_l1.shape[1], :, :].to(device), \
            pan_coefs2[0][:, pan_l1.shape[1]:, :, :].to(device)
        pan_coefs3, _ = self.__pdfbdec(pan_l2, nlevs=self.nlevs)
        pan_l3, pan_s3 = pan_coefs3[0][:, :pan_l2.shape[1], :, :].to(device), \
            pan_coefs3[0][:, pan_l2.shape[1]:, :, :].to(device)

        ms_coefs, _ = self.__pdfbdec(ms, nlevs=self.nlevs)
        ms_l, ms_s = ms_coefs[0][:, :ms.shape[1], :, :].to(device), ms_coefs[0][:, ms.shape[1]:, :, :].to(device)
        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), ms_l)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        # out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), ms_s)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p1(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p2(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p3(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        # m = self.conv_m(ms)  # [20, 64, 16, 16]
        # m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        # m_out = torch.mul(m_out, F.avg_pool2d(self.SA_m1(out_s[2]), 4))
        # m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        # m_out = torch.mul(m_out, F.avg_pool2d(self.SA_m2(out_s[1]), 4))
        # m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        # m_out = torch.mul(m_out, F.avg_pool2d(self.SA_m3(out_s[0]), 4))
        # m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        # mi = mutual_information(p_out, m_out)
        out = p_out
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out, 0


class CLSTM2(nn.Module):
    def __init__(self, block, num_blocks, args, max_channel=16,
                 variant="SSF", spec_type="all"):
        super(CLSTM2, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.nlevs = [0, 2, 2, 2]
        self.in_planes_1 = max_channel
        self.in_planes_2 = max_channel

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer_1(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer_1(block, max_channel, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer_1(block, max_channel, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer_1(block, max_channel, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer_2(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer_2(block, max_channel, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer_2(block, max_channel, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer_2(block, max_channel, num_blocks[3], stride=2)
        #
        # self.GRU_l = DH_CGRU(in_dim=1, h_dim=4, k_size=(3, 3), mode='SA')
        # self.GRU_s = DH_CGRU(in_dim=8, h_dim=32, k_size=(3, 3))

        self.GRU_s1 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l1 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))
        self.GRU_s2 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l2 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))
        self.GRU_s3 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l3 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.SA_p1 = SpatialAttention()
        self.SA_p2 = SpatialAttention()
        self.SA_p3 = SpatialAttention()
        self.SA_m1 = SpatialAttention()
        self.SA_m2 = SpatialAttention()
        self.SA_m3 = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * block.expansion, args['Categories_Number'])

    def _make_layer_1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_1, planes, stride))
            self.in_planes_1 = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_2, planes, stride))
            self.in_planes_2 = planes * block.expansion
        return nn.Sequential(*layers)

    def __pdfbdec(self, x, nlevs, method="resize"):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="9-7", dfilt="vk", nlevs=nlevs, device=device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, ms, pan):
        pan_coefs, _ = self.__pdfbdec(pan, self.nlevs)
        pan_s1, pan_s2, pan_s3 = pan_coefs[0].to(device), pan_coefs[1].to(device), pan_coefs[2].to(device)
        # torch.Size([20, 4, 32, 32]) torch.Size([20, 4, 16, 16]) torch.Size([20, 4, 8, 8])
        ms_coefs, _ = self.__pdfbdec(ms, [0, 2])
        ms_s = ms_coefs[0].to(device)  # torch.Size([20, 16, 8, 8])

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        m = self.conv_m(ms)  # [20, 64, 16, 16]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        s1, state_s1 = self.GRU_s1(pan_s1, m_out, state=0, mode='1')  # [20, 4, 32, 32], [20, 64, 32, 32]
        l1, state_l1 = self.GRU_l1(ms_s, p_out, state=0, mode='1')  # [20, 16, 8, 8], [20, 64, 8, 8]
        p_out = torch.mul(p_out, self.upsample4(self.SA_p1(l1)))
        m_out = torch.mul(m_out, F.avg_pool2d(self.SA_m1(s1), 4))

        p_out = self.p_layer2(p_out)  # [20, 64, 16, 16]
        m_out = self.m_layer2(m_out)  # [20, 64, 4, 4]
        s2, state_s2 = self.GRU_s2(pan_s2, m_out, state_s1, mode='2')  # [20, 4, 16, 16], [20, 64, 16, 16]
        l2, state_l2 = self.GRU_l2(ms_s, p_out, state_l1, mode='2')  # [20, 16, 8, 8], [20, 64, 8, 8]
        p_out = torch.mul(p_out, self.upsample2(self.SA_p2(l2)))  # [20, 64, 16, 16]
        m_out = torch.mul(m_out, F.avg_pool2d(self.SA_m2(s2), 4))

        p_out = self.p_layer3(p_out)  # [20, 64, 8, 8]
        m_out = self.m_layer3(m_out)  # [20, 64, 2, 2]
        s3, state_s3 = self.GRU_s3(pan_s3, m_out, state_s2, mode='2')  # [20, 4, 8, 8], [20, 64, 8, 8]
        l3, state_l3 = self.GRU_l3(ms_s, p_out, state_l2, mode='2')  # [20, 16, 8, 8], [20, 64, 8, 8]
        p_out = torch.mul(p_out, self.SA_p3(l3))  # [20, 64, 8, 8]
        m_out = torch.mul(m_out,  F.avg_pool2d(self.SA_m3(s3), 4))  # [20, 64, 2, 2]

        p_out = self.p_layer4(p_out)  # [20, 64, 4, 4]
        m_out = self.m_layer4(m_out)  # [20, 64, 1, 1]
        out = F.avg_pool2d(p_out, 4) + m_out
        out = out.view(out.size(0), -1)  # torch.Size([20, 512]torch.Size([20, 12]))
        out = self.linear(out)  #
        return out, 0


class CLSTM3(nn.Module):
    def __init__(self, block, num_blocks, args, max_channel=16,
                 variant="SSF", spec_type="all"):
        super(CLSTM3, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.nlevs = [0, 2, 2, 2]
        self.in_planes_1 = max_channel
        self.in_planes_2 = max_channel

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer_1(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer_1(block, max_channel, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer_1(block, max_channel, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer_1(block, max_channel, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer_2(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer_2(block, max_channel, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer_2(block, max_channel, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer_2(block, max_channel, num_blocks[3], stride=2)

        self.GRU_s1 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l1 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))
        self.GRU_s2 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l2 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))
        self.GRU_s3 = DH_CGRU_mini(in_dim=4, h_dim=max_channel, k_size=(3, 3))
        self.GRU_l3 = DH_CGRU_mini(in_dim=16, h_dim=max_channel, k_size=(3, 3))

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.SA_p1 = SpatialAttention()
        self.SA_p2 = SpatialAttention()
        self.SA_p3 = SpatialAttention()
        self.SA_m1 = SpatialAttention()
        self.SA_m2 = SpatialAttention()
        self.SA_m3 = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * block.expansion, args['Categories_Number'])

    def _make_layer_1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_1, planes, stride))
            self.in_planes_1 = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes_2, planes, stride))
            self.in_planes_2 = planes * block.expansion
        return nn.Sequential(*layers)

    def __pdfbdec(self, x, nlevs=[0, 3, 3, 3], method="resize"):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="9-7", dfilt="vk", nlevs=nlevs, device=device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, ms, pan):
        pan_coefs, _ = self.__pdfbdec(pan, self.nlevs)
        pan_s1, pan_s2, pan_s3 = pan_coefs[0].to(device), pan_coefs[1].to(device), pan_coefs[2].to(device)
        # torch.Size([20, 8, 32, 32]) torch.Size([20, 8, 16, 16]) torch.Size([20, 8, 8, 8])
        m = self.upsample(ms)  # torch.size([20, 4, 64, 64])
        ms_coefs, _ = self.__pdfbdec(m, self.nlevs)
        ms_s1, ms_s2, ms_s3 = ms_coefs[0].to(device), ms_coefs[1].to(device), ms_coefs[2].to(device)
        # torch.Size([20, 32, 32, 32]) torch.Size([20, 32, 16, 16]) torch.Size([20, 32, 8, 8])

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        m = self.conv_m(m)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        m_out = self.m_layer1(m)  # [20, 64, 32, 32]
        s1, state_s1 = self.GRU_s1(pan_s1, m_out, state=0, mode='1')  # [20, 8, 32, 32]
        l1, state_l1 = self.GRU_l1(ms_s1, p_out, state=0, mode='1')  # [20, 32, 32, 32]
        p_out = torch.mul(p_out, self.SA_p1(l1))
        m_out = torch.mul(m_out, self.SA_m1(s1))

        p_out = self.p_layer2(p_out)  # [20, 64, 16, 16]
        m_out = self.m_layer2(m_out)  # [20, 64, 16, 16]
        s2, state_s2 = self.GRU_s2(pan_s2, m_out, state_s1, mode='2')  # [20, 8, 16, 16]
        l2, state_l2 = self.GRU_l2(ms_s2, p_out, state_l1, mode='2')  # [20, 32, 16, 16]
        p_out = torch.mul(p_out, self.SA_p2(l2))  # [20, 32, 16, 16]
        m_out = torch.mul(m_out, self.SA_m2(s2))

        p_out = self.p_layer3(p_out)  # [20, 64, 8, 8]
        m_out = self.m_layer3(m_out)  # [20, 64, 8, 8]
        s3, state_s3 = self.GRU_s3(pan_s3, m_out, state_s2, mode='2')  # [20, 8, 8, 8]
        l3, state_l3 = self.GRU_l3(ms_s3, p_out, state_l2, mode='2')  # [20, 32, 8, 8]
        p_out = torch.mul(p_out, self.SA_p3(l3))  # [20, 32, 8, 8]
        m_out = torch.mul(m_out, self.SA_m3(s3))

        p_out = self.p_layer4(p_out)
        m_out = self.m_layer4(m_out)
        out = p_out + m_out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512]torch.Size([20, 12]))
        out = self.linear(out)  #
        return out, 0


class CLSTM4(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, variant="SSF", spec_type="all", save_address=''):
        super(CLSTM4, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.in_planes1 = 64
        self.in_planes2 = 64
        self.nlevs = [2]
        self.address = save_address

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer1(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer1(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer1(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer1(block, max_channel*8, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer2(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer2(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer2(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer2(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer2(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=1, h_dim=4, k_size=(3, 3), mode='SA')
        self.GRU_s = DH_CGRU_pro(in_dim=16, h_dim=4, k_size=(3, 3), mode='SA')
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes1, planes, stride))
            self.in_planes1 = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes2, planes, stride))
            self.in_planes2 = planes * block.expansion
        return nn.Sequential(*layers)

    def __pdfbdec(self, x, nlevs=[0, 3, 3, 3], method="resize"):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, nlevs=nlevs, pfilt="9-7", dfilt="vk", device=device)#  pfilt="9-7",

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        # CT module
        # pan_coefs, _ = self.__pdfbdec(pan, nlevs=[0, 2, 2, 2])
        # ms_coefs, _ = self.__pdfbdec(ms, nlevs=[0, 2, 2, 2])
        # pan_s1, pan_s2, pan_s3 = pan_coefs[0].to(device), pan_coefs[1].to(device), pan_coefs[2].to(device)
        # ms_s1, ms_s2, ms_s3 = ms_coefs[0].to(device), ms_coefs[1].to(device), ms_coefs[2].to(device)
        pan_coefs1, _ = self.__pdfbdec(pan, nlevs=self.nlevs)
        pan_l1, pan_s1 = pan_coefs1[0][:, :pan.shape[1], :, :].to(device), \
            pan_coefs1[0][:, pan.shape[1]:, :, :].to(device)
        pan_coefs2, _ = self.__pdfbdec(pan_l1, nlevs=self.nlevs)
        pan_l2, pan_s2 = pan_coefs2[0][:, :pan_l1.shape[1], :, :].to(device), \
            pan_coefs2[0][:, pan_l1.shape[1]:, :, :].to(device)
        pan_coefs3, _ = self.__pdfbdec(pan_l2, nlevs=self.nlevs)
        pan_l3, pan_s3 = pan_coefs3[0][:, :pan_l2.shape[1], :, :].to(device), \
            pan_coefs3[0][:, pan_l2.shape[1]:, :, :].to(device)
        # m1 = torch.unsqueeze(torch.mean(ms, dim=1), dim=1)
        ms_coefs1, _ = self.__pdfbdec(ms, nlevs=self.nlevs)
        ms_l1, ms_s1 = ms_coefs1[0][:, :ms.shape[1], :, :].to(device), \
            ms_coefs1[0][:, ms.shape[1]:, :, :].to(device)
        ms_coefs2, _ = self.__pdfbdec(ms_l1, nlevs=self.nlevs)
        ms_l2, ms_s2 = ms_coefs2[0][:, :ms_l1.shape[1], :, :].to(device), \
            ms_coefs2[0][:, ms_l1.shape[1]:, :, :].to(device)
        ms_coefs3, _ = self.__pdfbdec(ms_l2, nlevs=self.nlevs)
        ms_l3, ms_s3 = ms_coefs3[0][:, :ms_l2.shape[1], :, :].to(device), \
            ms_coefs3[0][:, ms_l2.shape[1]:, :, :].to(device)
        # [20, 4, 8, 8], [20, 16, 8, 8]
        # if self.address != '':
        #     if os.path.exists(self.address) == 0:
        #         os.makedirs(self.address)
        #
        #     save_image(ms, self.address+'ms.png', normalize=True)
        #     save_image(pan, self.address+'pan.png', normalize=True)
        #     save_image(pan_l1, self.address + "pan_l1.png", normalize=True)
        #     save_image(pan_l2, self.address + "pan_l2.png", normalize=True)
        #     save_image(pan_l3, self.address + "pan_l3.png", normalize=True)
        #     for i in range(pan_s1.shape[1]):
        #         save_image(torch.unsqueeze(pan_s1[:, i, :, :], dim=1), self.address + f"pan_s1{i}.png")
        #     for i in range(pan_s1.shape[1]-1):
        #         save_image(torch.unsqueeze(pan_s1[:, i+1, :, :] - pan_s1[:, i, :, :], dim=1),
        #                    self.address + f"pan_s1grad{i}.png")
        #     for i in range(pan_s2.shape[1]):
        #         save_image(torch.unsqueeze(pan_s2[:, i, :, :], dim=1), self.address + f"pan_s2{i}.png")
        #     for i in range(pan_s2.shape[1]-1):
        #         save_image(torch.unsqueeze(pan_s2[:, i+1, :, :] - pan_s2[:, i, :, :], dim=1),
        #                    self.address + f"pan_s2grad{i}.png")
        #     for i in range(pan_s3.shape[1]):
        #         save_image(torch.unsqueeze(pan_s3[:, i, :, :], dim=1), self.address + f"pan_s3{i}.png")
        #     for i in range(pan_s3.shape[1]-1):
        #         save_image(torch.unsqueeze(pan_s3[:, i+1, :, :] - pan_s3[:, i, :, :], dim=1),
        #                    self.address + f"pan_s3grad{i}.png")
        #     save_image(ms_l1, self.address + "ms_l1.png", normalize=True)
        #     save_image(ms_l2, self.address + "ms_l2.png", normalize=True)
        #     save_image(ms_l3, self.address + "ms_l3.png", normalize=True)
        #     for i in range(ms_s1.shape[1]):
        #         save_image(torch.unsqueeze(ms_s1[:, i, :, :], dim=1), self.address + f"ms_s1{i}.png", normalize=True)
        #     for i in range(ms_s2.shape[1]):
        #         save_image(torch.unsqueeze(ms_s2[:, i, :, :], dim=1), self.address + f"ms_s2{i}.png", normalize=True)
        #     for i in range(ms_s3.shape[1]):
        #         save_image(torch.unsqueeze(ms_s3[:, i, :, :], dim=1), self.address + f"ms_s3{i}.png", normalize=True)
        # RNN module
        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        m_out = F.avg_pool2d(m_out, 4)

        # mi = mutual_information(p_out, m_out)
        out = m_out + p_out
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class CLSTM5(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, variant="SSF", spec_type="all", save_address=''):
        super(CLSTM5, self).__init__()
        self.variant = variant
        self.spec_type = spec_type

        self.in_planes_2 = 64
        self.nlevs = [2]
        self.address = save_address
        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)

        self.in_planes = 64
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)
        self.in_planes = 64
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=1, h_dim=4, k_size=(1, 1), mode='SA')
        self.GRU_s = DH_CGRU_pro(in_dim=16, h_dim=4, k_size=(1, 1), mode='SA')
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = RevNetModel(num_channels=max_channel * 8, num_layers=6)
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

    def __pdfbdec(self, x, nlevs=[0, 3, 3, 3], method="resize"):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, nlevs=nlevs, pfilt="9-7", dfilt="vk", device=device)#  pfilt="9-7",

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, m, pan):
        ms = self.upsample(m)
        # ms:[20, 4, 64, 64] pan:[20, 1, 64, 64]
        pan_coefs1, _ = self.__pdfbdec(pan, nlevs=self.nlevs)
        pan_l1, pan_s1 = pan_coefs1[0][:, :pan.shape[1], :, :].to(device), \
            pan_coefs1[0][:, pan.shape[1]:, :, :].to(device)
        pan_coefs2, _ = self.__pdfbdec(pan_l1, nlevs=self.nlevs)
        pan_l2, pan_s2 = pan_coefs2[0][:, :pan_l1.shape[1], :, :].to(device), \
            pan_coefs2[0][:, pan_l1.shape[1]:, :, :].to(device)
        pan_coefs3, _ = self.__pdfbdec(pan_l2, nlevs=self.nlevs)
        pan_l3, pan_s3 = pan_coefs3[0][:, :pan_l2.shape[1], :, :].to(device), \
            pan_coefs3[0][:, pan_l2.shape[1]:, :, :].to(device)

        # m1 = torch.unsqueeze(torch.mean(ms, dim=1), dim=1)
        ms_coefs1, _ = self.__pdfbdec(ms, nlevs=self.nlevs)
        ms_l1, ms_s1 = ms_coefs1[0][:, :ms.shape[1], :, :].to(device), \
            ms_coefs1[0][:, ms.shape[1]:, :, :].to(device)
        ms_coefs2, _ = self.__pdfbdec(ms_l1, nlevs=self.nlevs)
        ms_l2, ms_s2 = ms_coefs2[0][:, :ms_l1.shape[1], :, :].to(device), \
            ms_coefs2[0][:, ms_l1.shape[1]:, :, :].to(device)
        ms_coefs3, _ = self.__pdfbdec(ms_l2, nlevs=self.nlevs)
        ms_l3, ms_s3 = ms_coefs3[0][:, :ms_l2.shape[1], :, :].to(device), \
            ms_coefs3[0][:, ms_l2.shape[1]:, :, :].to(device)
        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        out = self.inn(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(out1 + out2, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class CLSTM6(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64):
        super(CLSTM6, self).__init__()
        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)

        self.in_planes = 64
        self.m_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)
        self.in_planes = 64
        self.p_layer1 = self._make_layer(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=1, h_dim=4, k_size=(1, 1), mode='SA')
        self.GRU_s = DH_CGRU_pro(in_dim=16, h_dim=4, k_size=(1, 1), mode='SA')
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.inn = RevNetModel(num_channels=max_channel * 8, num_layers=6)
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

    def forward(self, m, pan, ms_coefs, pan_coefs):
        ms = self.upsample(m)
        pan_l1, pan_s1 = pan_coefs[0][0][:, :pan.shape[1], :, :].to(device), \
            pan_coefs[0][0][:, pan.shape[1]:, :, :].to(device)
        pan_l2, pan_s2 = pan_coefs[1][0][:, :pan_l1.shape[1], :, :].to(device), \
            pan_coefs[1][0][:, pan_l1.shape[1]:, :, :].to(device)
        pan_l3, pan_s3 = pan_coefs[2][0][:, :pan_l2.shape[1], :, :].to(device), \
            pan_coefs[2][0][:, pan_l2.shape[1]:, :, :].to(device)
        ms_l1, ms_s1 = ms_coefs[0][0][:, :ms.shape[1], :, :].to(device), \
            ms_coefs[0][0][:, ms.shape[1]:, :, :].to(device)
        ms_l2, ms_s2 = ms_coefs[1][0][:, :ms_l1.shape[1], :, :].to(device), \
            ms_coefs[1][0][:, ms_l1.shape[1]:, :, :].to(device)
        ms_l3, ms_s3 = ms_coefs[2][0][:, :ms_l2.shape[1], :, :].to(device), \
            ms_coefs[2][0][:, ms_l2.shape[1]:, :, :].to(device)

        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        # p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        # m_out = F.avg_pool2d(m_out, 4)

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        # print(p_out.shape, m_out.shape)
        out = self.inn(torch.cat((p_out, m_out), dim=1))
        out1, out2 = torch.chunk(out, 2, dim=1)
        out = F.avg_pool2d(out1 + out2, 4)
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out


class CLSTM7(nn.Module):
    def __init__(self, block, num_blocks, args,
                 max_channel=64, variant="SSF", spec_type="all", save_address=''):
        super(CLSTM7, self).__init__()
        self.variant = variant
        self.spec_type = spec_type
        self.in_planes1 = 64
        self.in_planes2 = 64
        self.nlevs = 2
        self.address = save_address

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=max_channel, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=max_channel, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer1(block, max_channel, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer1(block, max_channel*2, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer1(block, max_channel*4, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer1(block, max_channel*8, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer2(block, max_channel, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer2(block, max_channel*2, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer2(block, max_channel*4, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer2(block, max_channel*8, num_blocks[3], stride=2)

        self.layer5 = self._make_layer2(block, max_channel * 8, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU_pro(in_dim=1, h_dim=4, k_size=(3, 3), mode='SA')
        self.GRU_s = DH_CGRU_pro(in_dim=16, h_dim=4, k_size=(3, 3), mode='SA')
        # self.GRU_s = DH_CGRU_s(in_dim=4*(2**self.nlevs[0]), h_dim=1*(2**self.nlevs[0]), k_size=(3, 3))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.SA_p = SpatialAttention()
        self.SA_m = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(max_channel * 8 * block.expansion, args['Categories_Number'])

    def _make_layer1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes1, planes, stride))
            self.in_planes1 = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes2, planes, stride))
            self.in_planes2 = planes * block.expansion
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

        # [20, 4, 8, 8], [20, 16, 8, 8]
        # RNN module
        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), (ms_l1, ms_l2, ms_l3))
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32], [20, 4, 8, 8]
        out_s, state_s = self.GRU_s((ms_s1, ms_s2, ms_s3), (pan_s1, pan_s2, pan_s3))
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, self.SA_p(out_l[2]))  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, self.SA_p(out_l[1]))
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, self.SA_p(out_l[0]))
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, self.SA_m(out_s[2]))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, self.SA_m(out_s[1]))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, self.SA_m(out_s[0]))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]
        m_out = F.avg_pool2d(m_out, 4)

        # mi = mutual_information(p_out, m_out)
        out = m_out + p_out
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out



class CT_trans(nn.Module):
    def __init__(self, nlevs=[0, 3, 3, 3], method='resize', variant="SSF",
                 spec_type="all", pfilt="maxflat", dfilt="dmaxflat7", device=torch.device("cpu")):
        super(CT_trans, self).__init__()
        self.nlevs = nlevs
        self.variant = variant
        self.method = method
        self.spec_type = spec_type
        self.pfilt = pfilt
        self.dfilt = dfilt
        self.device = device
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def __pdfbdec(self, x):
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, nlevs=self.nlevs, pfilt=self.pfilt,
                                            dfilt=self.dfilt, device=device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if self.method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs  # 低频信息在前面

    def forward(self, ms, pan):
        ms = self.upsample(ms)
        pan_coefs1, _ = self.__pdfbdec(pan)
        pan_l1, pan_s1 = pan_coefs1[0][:, :pan.shape[1], :, :].to(device), \
            pan_coefs1[0][:, pan.shape[1]:, :, :].to(device)
        pan_coefs2, _ = self.__pdfbdec(pan_l1)
        pan_l2, pan_s2 = pan_coefs2[0][:, :pan_l1.shape[1], :, :].to(device), \
            pan_coefs2[0][:, pan_l1.shape[1]:, :, :].to(device)
        pan_coefs3, _ = self.__pdfbdec(pan_l2)
        pan_l3, pan_s3 = pan_coefs3[0][:, :pan_l2.shape[1], :, :].to(device), \
            pan_coefs3[0][:, pan_l2.shape[1]:, :, :].to(device)
        # m1 = torch.unsqueeze(torch.mean(ms, dim=1), dim=1)
        ms_coefs1, _ = self.__pdfbdec(ms)
        ms_l1, ms_s1 = ms_coefs1[0][:, :ms.shape[1], :, :].to(device), \
            ms_coefs1[0][:, ms.shape[1]:, :, :].to(device)
        ms_coefs2, _ = self.__pdfbdec(ms_l1)
        ms_l2, ms_s2 = ms_coefs2[0][:, :ms_l1.shape[1], :, :].to(device), \
            ms_coefs2[0][:, ms_l1.shape[1]:, :, :].to(device)
        ms_coefs3, _ = self.__pdfbdec(ms_l2)
        ms_l3, ms_s3 = ms_coefs3[0][:, :ms_l2.shape[1], :, :].to(device), \
            ms_coefs3[0][:, ms_l2.shape[1]:, :, :].to(device)
        return (ms_coefs1, ms_coefs2, ms_coefs3), (pan_coefs1, pan_coefs2, pan_coefs3)


def CLSTM_1(cfg):
    return CLSTM1(BasicBlk, [1, 1, 1, 1], cfg)


def CLSTM_2():
    return CLSTM2(BasicBlk, [1, 1, 1, 1])


def CLSTM_3():
    return CLSTM3(BasicBlk, [1, 1, 1, 1])


def CLSTM_4(address=''):
    return CLSTM4(BasicBlk, [1, 1, 1, 1], save_address=address)


def CLSTM_5(address=''):
    return CLSTM5(BasicBlk, [1, 1, 1, 1], save_address=address)


def CLSTM_6():
    return CLSTM6(BasicBlk, [1, 1, 1, 1])


def CLSTM_7():
    return CLSTM7(BasicBlk, [1, 1, 1, 1])


def Net(args):
    return CLSTM8(BasicBlk, [1, 1, 1, 1], args)


def test_net():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8
    }
    # CT_4 = CT_transform(4)
    # CT_1 = CT_transform(1)
    # ms_l, ms_s = CT_4.contourlet_decompose(ms)
    # pan_l1, pan_s1 = CT_1.contourlet_decompose(pan)
    # pan_l2, pan_s2 = CT_1.contourlet_decompose(pan_l1)
    # pan_l3, pan_s3 = CT_1.contourlet_decompose(pan_l2)
    net = Net(cfg).to(device)
    # for i in range(10):
    #     time1 = time.time()
    #     y = net(ms, pan)
    #     print(time.time() - time1)
    y = net(ms, pan)
    print(y.shape)
    # print(y.shape, l)


def test_2():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    CT = CT_trans(nlevs=[2], pfilt='9-7', dfilt='vk', device=device)
    time1 = time.time()
    ms_coefs, pan_coefs = CT(ms, pan)
    print(time.time() - time1)
    net = CLSTM_6().to(device)
    y = net(ms, pan, ms_coefs, pan_coefs)
    print(time.time() - time1)
    print(y.shape)

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
        plt.title('Single Channel')
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
    plt.show()


if __name__ == '__main__':
    test_net()
    # import cv2
    # import numpy as np
    # from function.process_function.process_function import to_tensor
    #
    # img = cv2.imread('contourlet/image/barbara.png')
    # img = np.mean(img, axis=2)
    # img = to_tensor(img)
    # import matplotlib.pyplot as plt
    # # plt.imshow(img)
    # # plt.show()
    # # img = np.transpose(img, axes=(2, 0, 1))
    # img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=0)
    # print(img.shape)
    # img = torch.from_numpy(img).type(torch.FloatTensor)
    # coefs = batch_multi_channel_pdfbdec(x=img, nlevs=[3], device=device)  # pfilt="9-7", dfilt="vk",
    # method = 'resize'
    # # Stack channels with same image dimension
    # coefs = stack_same_dim(coefs)
    #
    # if method == "resize":
    #     for k in coefs.keys():
    #         # Resize if image is not square
    #         if k[2] != k[3]:
    #             # Get maximum dimension (height or width)
    #             max_dim = int(np.max((k[2], k[3])))
    #             # Resize the channels
    #             trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
    #             coefs[k] = trans(coefs[k])
    # else:
    #     for k in coefs.keys():
    #         # Resize if image is not square
    #         if k[2] != k[3]:
    #             # Get minimum dimension (height or width)
    #             min_dim = int(np.argmin((k[2], k[3]))) + 2
    #             # Splice alternate channels (always even number of channels exist)
    #             coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)
    #
    # # Stack channels with same image dimension
    # coefs = stack_same_dim(coefs)
    #
    # # Change coefs's key to number (n-1 to 0), instead of dimension
    # for i, k in enumerate(coefs.copy()):
    #     idx = len(coefs.keys()) - i - 1
    #     coefs[idx] = coefs.pop(k)
    # j = 0
    # for key in coefs:
    #     print(key, coefs[key].shape)
    #     for i in range(coefs[key].shape[1]):
    #         save_image(torch.unsqueeze(coefs[key][:, i, :, :], dim=1), f"s{j}{i}.png")#, normalization=True
    #     j = j + 1