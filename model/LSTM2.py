from resnet import conv3x3, BasicBlk
from LSTM import CT_transform, ChannelAttention, SpatialAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import CT
import config
import time
Categories = config.Categories
device = config.DEVICE


def denoise(ms, pan):
    m_u, m_sum, m_v = torch.linalg.svd(ms)
    p_u, p_sum, p_v = torch.linalg.svd(pan)
    sim = torch.sigmoid(torch.cosine_similarity(m_u, p_u, dim=1))
    sim = torch.unsqueeze(sim, dim=1)

    f_m_u = torch.mul(sim, m_u)
    f_p_u = torch.mul(1 - sim, p_u)
    f_m = torch.matmul(torch.matmul(f_m_u, torch.diag_embed(m_sum)), m_v)
    f_p = torch.matmul(torch.matmul(f_p_u, torch.diag_embed(m_sum)), p_v)
    return f_m, f_p


class H_RNN(nn.Module):
    def __init__(self, input_dim, block, num_blocks=5):
        super(H_RNN, self).__init__()
        self.in_planes = input_dim
        self.hidden_dim = 20
        self.cell = ConvGRUCell(self.in_planes, self.hidden_dim)
        self.layer = self._make_layer(block, num_blocks)

    def _make_layer(self, block, num_blocks):
        layers = []
        for num in range(num_blocks-1):
            layers.append(block(self.in_planes, self.hidden_dim))
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        tensor = input_tensor
        init_state = self.cell.init_hidden(tensor)
        out = self.cell([tensor, init_state])
        out = self.layer(out)
        return out


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = False
        self.conv_xz = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_hz = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_xr = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_hr = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_xh = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_hh = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size,
                                 stride=(1, 1), padding=self.padding, bias=self.bias)
        self.conv_out = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=self.kernel_size,
                                  stride=(1, 1), padding=self.padding, bias=self.bias)

    def init_hidden(self, tensor):
        batch_size = tensor.shape[0]
        height, width = tensor.shape[2], tensor.shape[3]
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, input_tensor, cur_state):
        z_t = torch.sigmoid(self.conv_xz(input_tensor) + self.conv_hz(cur_state))
        r_t = torch.sigmoid(self.conv_xr(input_tensor) + self.conv_hr(cur_state))
        h_hat_t = torch.tanh(self.conv_xh(input_tensor) + self.conv_hh(torch.mul(r_t, cur_state)))
        h_t = torch.mul((1 - z_t), cur_state) + torch.mul(z_t, h_hat_t)
        out = self.conv_out(h_t)
        return h_t, out


class H_CGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(H_CGRU, self).__init__()
        self.cell1 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell2 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell3 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell4 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell5 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.h_SA1 = SpatialAttention()
        self.h_CA1 = ChannelAttention(hidden_dim)
        self.h_CA2 = ChannelAttention(hidden_dim)
        self.h_CA3 = ChannelAttention(hidden_dim)
        self.h_CA4 = ChannelAttention(hidden_dim)
        self.h_CA5 = ChannelAttention(hidden_dim)
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_tensor, state=()):
        if state == ():
            init_state = self.cell1.init_hidden(input_tensor)
            cur_state1, out = self.cell1(input_tensor, init_state)
            cur_state2, out = self.cell2(out + input_tensor, cur_state1)
            cur_state3, out = self.cell3(out + input_tensor, cur_state2)
            cur_state4, out = self.cell4(out + input_tensor, cur_state3)
            cur_state5, out = self.cell5(out + input_tensor, cur_state4)
        else:
            init_state = self.cell1.init_hidden(input_tensor)
            cur_state1, out = self.cell1(input_tensor, torch.mul(init_state, self.h_CA1(state[0])))
            cur_state2, out = self.cell2(out + input_tensor, torch.mul(cur_state1, self.h_CA2(state[1])))
            cur_state3, out = self.cell3(out + input_tensor, torch.mul(cur_state1, self.h_CA3(state[2])))
            cur_state4, out = self.cell4(out + input_tensor, torch.mul(cur_state1, self.h_CA4(state[3])))
            cur_state5, out = self.cell5(out + input_tensor, torch.mul(cur_state1, self.h_CA5(state[4])))
        return out, (cur_state1, cur_state2, cur_state3, cur_state4, cur_state5)


class DH_CGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, level=5):
        super(DH_CGRU, self).__init__()
        self.cell1 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell2 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.cell3 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.hidden_cell1 = H_CGRU(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                   kernel_size=kernel_size, bias=bias)
        self.hidden_cell2 = H_CGRU(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                   kernel_size=kernel_size, bias=bias)
        self.hidden_cell3 = H_CGRU(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                   kernel_size=kernel_size, bias=bias)
        self.h_SA1 = SpatialAttention()
        self.h_SA2 = SpatialAttention()
        self.h_CA1 = ChannelAttention(hidden_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_tensor):
        init_state1 = self.cell1.init_hidden(input_tensor[2])
        cur_state1, out1 = self.cell1(input_tensor[2], init_state1)
        out_h1, h_state1 = self.hidden_cell1(cur_state1)
        init_state2 = self.cell2.init_hidden(input_tensor[1])
        cur_state2, out2 = self.cell2(input_tensor[1],
                                      torch.mul(init_state2, self.h_SA1(self.upsample(cur_state1))))
        out_h2, h_state2 = self.hidden_cell2(cur_state2)

        init_state3 = self.cell2.init_hidden(input_tensor[0])
        cur_state3, out3 = self.cell3(input_tensor[0],
                                      torch.mul(init_state3, self.h_SA2(self.upsample(cur_state2))))
        out_h3, h_state3 = self.hidden_cell3(cur_state3)
        return (out1, out2, out3), (h_state1, h_state2, h_state3)


class CT_LSTM_8(nn.Module):  # 降重模块后置
    def __init__(self, block=BasicBlk, num_blocks=(1, 1, 1, 1), num_classes=Categories):
        super(CT_LSTM_8, self).__init__()
        self.in_planes = 64

        self.c_p_l = nn.Conv2d(1, 16, (1, 1), (1, 1), bias=False)
        self.c_p_s = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_l = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_s = nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)

        self.CGRU_p_s = DH_CGRU(4, 20, (3, 3), False)
        self.CGRU_p_l = DH_CGRU(1, 20, (3, 3), False)

        self.CGRU_m_s = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_m_l = H_CGRU(4, 20, (3, 3), False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.downsample_s1 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.downsample_s2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.downsample_l1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downsample_l2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.conv_downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.trans = nn.Conv2d(20, 5, (1, 1), (1, 1), bias=False)
        self.trans2 = nn.Conv2d(5, 64, (1, 1), (1, 1), bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.ct_m = CT_transform(4)
        self.ct_p = CT_transform(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def denoise(self, ms, pan):
        m_u, m_sum, m_v = torch.linalg.svd(ms)
        p_u, p_sum, p_v = torch.linalg.svd(pan)
        sim = torch.sigmoid(torch.cosine_similarity(m_u, p_u, dim=1))
        sim = torch.unsqueeze(sim, dim=1)

        f_m_u = torch.mul(sim, m_u)
        f_p_u = torch.mul(1 - sim, p_u)
        f_m = torch.matmul(torch.matmul(f_m_u, torch.diag_embed(m_sum)), m_v)
        f_p = torch.matmul(torch.matmul(f_p_u, torch.diag_embed(m_sum)), p_v)
        return f_m, f_p

    def forward(self, ms, pan):
        # CT模块
        m_l1, m_s1 = self.ct_m.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])

        p_l1, p_s1 = self.ct_p.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])
        p_l2, p_s2 = self.ct_p.contourlet_decompose(p_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])
        p_l3, p_s3 = self.ct_p.contourlet_decompose(p_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])

        # RNN模块
        p_out_s, p_state_s = self.CGRU_p_s((p_s1, p_s2, p_s3))
        # torch.Size([20, 4, 8, 8]) torch.Size([20, 4, 16, 16]) torch.Size([20, 4, 32, 32])
        p_out_l, p_state_l = self.CGRU_p_l((p_l1, p_l2, p_l3))
        # torch.Size([20, 1, 8, 8]) torch.Size([20, 1, 16, 16]) torch.Size([20, 1, 32, 32])

        m_out_s, m_state_s = self.CGRU_m_s(m_s1, p_state_s[0])  # torch.Size([20, 16, 8, 8])
        m_out_l, m_state_l = self.CGRU_m_l(m_l1, p_state_l[0])  # torch.Size([20, 4, 8, 8])

        p_scale_s = self.downsample_s1(self.downsample_s2(p_out_s[2])+p_out_s[1]) + p_out_s[0]
        # torch.Size([20, 4, 8, 8])
        p_scale_l = self.downsample_l1(self.downsample_l2(p_out_l[2])+p_out_l[1]) + p_out_l[0]
        # torch.Size([20, 1, 8, 8])

        p = torch.concat((p_scale_s, p_scale_l), dim=1)  # torch.Size([20, 5, 8, 8])
        m = torch.concat((m_out_s, m_out_l), dim=1)  # torch.Size([20, 20, 8, 8])
        m = self.trans(m)  # torch.Size([20, 20, 8, 8])

        m1, p1 = self.denoise(m, p)
        # MSE = nn.MSELoss()
        # loss = MSE(m1, m) + MSE(p1, p)
        out = m1 + p1  # torch.Size([20, 40, 8, 8])
        out = self.trans2(out)  # torch.Size([20, 64, 8, 8])
        out = self.layer1(out)  # torch.Size([20, 64, 8, 8])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        # out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])

        return out, 1


class CT_LSTM_9(nn.Module):  # 降重模块后置
    def __init__(self, block=BasicBlk, num_blocks=(1, 1, 1, 1), num_classes=Categories):
        super(CT_LSTM_9, self).__init__()
        self.in_planes = 64
        self.conv_m1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)

        self.conv_p1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv_p2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.conv_p3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)

        self.CGRU_p_s = DH_CGRU(4, 20, (3, 3), False)
        self.CGRU_p_l = DH_CGRU(1, 20, (3, 3), False)

        self.CGRU_m_s = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_m_l = H_CGRU(4, 20, (3, 3), False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.downsample_s1 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.downsample_s2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.downsample_l1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downsample_l2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.conv_downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.trans = nn.Conv2d(20, 5, (1, 1), (1, 1), bias=False)
        self.trans2 = nn.Conv2d(5, 64, (1, 1), (1, 1), bias=False)
        self.trans3 = nn.Conv2d(40, 64, (1, 1), (1, 1), bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.ct_m = CT_transform(4)
        self.ct_p = CT_transform(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def denoise(self, ms, pan):
        m_u, m_sum, m_v = torch.linalg.svd(ms)
        p_u, p_sum, p_v = torch.linalg.svd(pan)
        sim = torch.sigmoid(torch.cosine_similarity(m_u, p_u, dim=1))
        sim = torch.unsqueeze(sim, dim=1)

        f_m_u = torch.mul(sim, m_u)
        f_p_u = torch.mul(1 - sim, p_u)
        f_m = torch.matmul(torch.matmul(f_m_u, torch.diag_embed(m_sum)), m_v)
        f_p = torch.matmul(torch.matmul(f_p_u, torch.diag_embed(m_sum)), p_v)
        return f_m, f_p

    def forward(self, ms, pan):  # torch.size([20, 4, 16, 16]), torch.size([20, 1, 64, 64])
        # CT模块
        m_l1, m_s1 = self.ct_m.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        m_stem = torch.concat((self.conv_m1(ms), m_s1), dim=1)  # torch.Size([20, 32, 8, 8])

        p_l1, p_s1 = self.ct_p.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])
        p_stem1 = torch.concat((self.conv_p1(pan), p_s1), dim=1)  # torch.Size([20, 8, 32, 32])
        p_l2, p_s2 = self.ct_p.contourlet_decompose(p_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])
        p_stem2 = torch.concat((self.conv_p2(p_stem1), p_s2), dim=1)  # torch.Size([20, 8, 16, 16])
        p_l3, p_s3 = self.ct_p.contourlet_decompose(p_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        p_stem3 = torch.concat((self.conv_p3(p_stem2), p_s3), dim=1)  # torch.Size([20, 8, 8, 8])

        out1 = self.trans3(torch.concat((p_stem3, m_stem), dim=1))  # torch.Size([20, 64, 8, 8])
        # RNN模块
        p_out_s, p_state_s = self.CGRU_p_s((p_s1, p_s2, p_s3))
        # torch.Size([20, 4, 8, 8]) torch.Size([20, 4, 16, 16]) torch.Size([20, 4, 32, 32])
        p_out_l, p_state_l = self.CGRU_p_l((p_l1, p_l2, p_l3))
        # torch.Size([20, 1, 8, 8]) torch.Size([20, 1, 16, 16]) torch.Size([20, 1, 32, 32])

        m_out_s, m_state_s = self.CGRU_m_s(m_s1, p_state_s[0])  # torch.Size([20, 16, 8, 8])
        m_out_l, m_state_l = self.CGRU_m_l(m_l1, p_state_l[0])  # torch.Size([20, 4, 8, 8])

        p_scale_s = self.downsample_s1(self.downsample_s2(p_out_s[2])+p_out_s[1]) + p_out_s[0]
        # torch.Size([20, 4, 8, 8])
        p_scale_l = self.downsample_l1(self.downsample_l2(p_out_l[2])+p_out_l[1]) + p_out_l[0]
        # torch.Size([20, 1, 8, 8])

        p = torch.concat((p_scale_s, p_scale_l), dim=1)  # torch.Size([20, 5, 8, 8])
        m = torch.concat((m_out_s, m_out_l), dim=1)  # torch.Size([20, 20, 8, 8])
        m = self.trans(m)  # torch.Size([20, 20, 8, 8])

        m1, p1 = self.denoise(m, p)
        # MSE = nn.MSELoss()
        # loss = MSE(m1, m) + MSE(p1, p)
        out = m1 + p1  # torch.Size([20, 40, 8, 8])
        out = self.trans2(out)  # torch.Size([20, 64, 8, 8])
        out = out + out1
        out = self.layer1(out)  # torch.Size([20, 64, 8, 8])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        # out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])

        return out, 1


def CT_10_8():
    return CT_LSTM_8(BasicBlk, [1, 1, 1, 1])


def CT_10_9():
    return CT_LSTM_9(BasicBlk, [1, 1, 1, 1])


def test_net():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    # CT_4 = CT_transform(4)
    # CT_1 = CT_transform(1)
    # ms_l, ms_s = CT_4.contourlet_decompose(ms)
    # pan_l1, pan_s1 = CT_1.contourlet_decompose(pan)
    # pan_l2, pan_s2 = CT_1.contourlet_decompose(pan_l1)
    # pan_l3, pan_s3 = CT_1.contourlet_decompose(pan_l2)
    net = CT_10_9().to(device)
    y, l = net(ms, pan)
    # print(y.shape, l)


if __name__ == '__main__':
    test_net()
