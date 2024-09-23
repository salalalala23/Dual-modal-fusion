from .resnet import BasicBlk
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import CT
from .LSTM import ChannelAttention, SpatialAttention
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mutual_information(x, y):
    # 计算x和y的联合分布
    xy = torch.einsum('bchw,bchw->bc', x, y)
    xy = torch.sigmoid(F.normalize(xy, p=1, dim=1))
    # 计算x和y的边缘分布
    x_marginal = torch.mean(x, dim=[2, 3], keepdim=True)
    y_marginal = torch.mean(y, dim=[2, 3], keepdim=True)
    x_marginal = torch.sigmoid(F.normalize(x_marginal.view(x_marginal.size(0), -1), dim=1))
    y_marginal = torch.sigmoid(F.normalize(y_marginal.view(y_marginal.size(0), -1), dim=1))
    # 计算互信息
    mi = torch.einsum('bc,bc->b', xy, torch.log(xy / (x_marginal * y_marginal)))
    return mi.mean()


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
        self.SACA = SACA(h_dim, name)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def init_hidden(self, tensor):
        batch_size = tensor.shape[0]
        height, width = tensor.shape[2], tensor.shape[3]
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, in_tensor, state, mode):
        if mode == '1':
            out, cur_state1 = self.cell1(in_tensor, self.SACA(state))
            out, cur_state2 = self.cell2(out+in_tensor, self.SACA(cur_state1))
            out, cur_state3 = self.cell3(out+in_tensor, self.SACA(cur_state2))
        else:
            if state[0].shape[3] != in_tensor.shape[3]:
                out, cur_state1 = self.cell1(in_tensor, self.SACA(self.upsample(state[0])))
                out, cur_state2 = self.cell2(out+in_tensor, cur_state1+self.SACA(self.upsample(state[1])))
                out, cur_state3 = self.cell3(out+in_tensor, cur_state2+self.SACA(self.upsample(state[2])))
            else:
                out, cur_state1 = self.cell1(in_tensor, self.SACA(state[0]))
                out, cur_state2 = self.cell2(out + in_tensor, cur_state1+self.SACA(state[1]))
                out, cur_state3 = self.cell3(out + in_tensor, cur_state2+self.SACA(state[2]))
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
        out_h1, h_state1 = self.h_cell1(out1, cur_state1, mode='1')

        scale2 = self.upsample(scale1)  # (20, 1, 16, 16)
        out2, cur_state2 = self.cell2(in_tensor1[1], scale2)
        out_h2, h_state2 = self.h_cell2(out2, h_state1, mode='2')

        scale3 = self.upsample(scale2)  # (20, 1, 32, 32)
        out3, cur_state3 = self.cell3(in_tensor1[0], scale3)
        out_h3, h_state3 = self.h_cell3(out3, h_state2, mode='2')

        return (out_h1, out_h2, out_h3), (h_state1, h_state2, h_state3)


class CT_LSTM_10(nn.Module):  # 降重模块后置
    def __init__(self, block, num_blocks, args):
        super(CT_LSTM_10, self).__init__()
        self.in_planes_1 = 64
        self.in_planes_2 = 64

        self.conv_m = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)
        self.m_layer1 = self._make_layer_1(block, 64, num_blocks[0], stride=2)
        self.m_layer2 = self._make_layer_1(block, 128, num_blocks[1], stride=2)
        self.m_layer3 = self._make_layer_1(block, 256, num_blocks[2], stride=2)
        self.m_layer4 = self._make_layer_1(block, 512, num_blocks[3], stride=2)

        self.p_layer1 = self._make_layer_2(block, 64, num_blocks[0], stride=2)
        self.p_layer2 = self._make_layer_2(block, 128, num_blocks[1], stride=2)
        self.p_layer3 = self._make_layer_2(block, 256, num_blocks[2], stride=2)
        self.p_layer4 = self._make_layer_2(block, 512, num_blocks[3], stride=2)

        self.GRU_l = DH_CGRU(in_dim=1, h_dim=4, k_size=(3, 3), mode='SA')
        self.GRU_s = DH_CGRU(in_dim=4, h_dim=16, k_size=(3, 3))

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(512 * block.expansion, args['Categories_Number'])
        self.linear2 = nn.Linear(512 * block.expansion, args['Categories_Number'])

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

    def forward(self, ms, pan):
        # ms:[20, 4, 16, 16] pan:[20, 1, 64, 64]
        # CT module
        ms_l, ms_s = CT.contourlet_decompose(ms)  # [20, 4, 8, 8], [20, 16, 8, 8]
        pan_l1, pan_s1 = CT.contourlet_decompose(pan)  # [20, 1, 32, 32], [20, 4, 32, 32]
        pan_l2, pan_s2 = CT.contourlet_decompose(pan_l1)  # [20, 1, 16, 16], [20, 4, 16, 16]
        pan_l3, pan_s3 = CT.contourlet_decompose(pan_l2)  # [20, 1, 8, 8], [20, 4, 8, 8]

        # RNN module
        out_l, state_l = self.GRU_l((pan_l1, pan_l2, pan_l3), ms_l)
        # [20, 1, 8, 8], [20, 1, 16, 16], [20, 1, 32, 32]
        out_s, state_s = self.GRU_s((pan_s1, pan_s2, pan_s3), ms_s)
        # [20, 4, 8, 8], [20, 4, 16, 16], [20, 4, 32, 32]

        p = self.conv_p(pan)  # [20, 64, 64, 64]
        p_out = self.p_layer1(p)  # [20, 64, 32, 32]
        p_out = torch.mul(p_out, out_l[2])  # [20, 64, 32, 32]
        p_out = self.p_layer2(p_out)  # [20, 128, 16, 16]
        p_out = torch.mul(p_out, out_l[1])
        p_out = self.p_layer3(p_out)  # [20, 256, 8, 8]
        p_out = torch.mul(p_out, out_l[0])
        p_out = self.p_layer4(p_out)  # torch.Size([20, 512, 4, 4])
        p_out = F.avg_pool2d(p_out, 4)  # torch.Size([20, 512, 1, 1])

        m = self.conv_m(ms)  # [20, 64, 16, 16]
        m_out = self.m_layer1(m)  # [20, 64, 8, 8]
        m_out = torch.mul(m_out, F.avg_pool2d(self.SA1(out_s[2]), 4))
        m_out = self.m_layer2(m_out)  # [20, 128, 4, 4]
        m_out = torch.mul(m_out, F.avg_pool2d(self.SA2(out_s[1]), 4))
        m_out = self.m_layer3(m_out)  # [20, 256, 2, 2]
        m_out = torch.mul(m_out, F.avg_pool2d(self.SA3(out_s[0]), 4))
        m_out = self.m_layer4(m_out)  # [20, 512, 1， 1]

        # mi = mutual_information(p_out, m_out)
        # out = p_out + m_out
        out1 = p_out.view(p_out.size(0), -1)  # torch.Size([20, 512])
        out2 = m_out.view(m_out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear1(out1) + self.linear2(out2)  # torch.Size([20, 12])
        return out, 0
        # m = self.conv_m(ms)  # [20, 16, 16, 16]

        # scale_s = torch.cat((out_l[0], out_s[0]), dim=1)  # [20, 5, 8, 8]
        # scale_m = torch.cat((out_l[1], out_s[1]), dim=1)  # [20, 5, 16, 16]
        # scale_l = torch.cat((out_l[2], out_s[2]), dim=1)  # [20, 5, 32, 32]


def CT_10_10():
    return CT_LSTM_10(BasicBlk, [1, 1, 1, 1])  # 验证DHCGRU顺序的影响

#
# def CT_10_11():
#     return CT_LSTM_11(BasicBlk, [1, 1, 1, 1])
#
#
# def CT_10_12():
#     return CT_LSTM_12(BasicBlk, [1, 1, 1, 1])
#
#
# def CT_10_13():
#     return CT_LSTM_13(BasicBlk, [1, 1, 1, 1])

def test_CGRU():
    x = torch.randn([2, 4, 16, 16]).to(device)
    y = torch.randn([2, 20, 16, 16]).to(device)
    net = DH_CGRU(4, 20, [3, 3]).to(device)
    h, y = net(x, y)
    print(h[0].shape, y[0][0].shape)


def test_net():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    # CT_4 = CT_transform(4)
    # CT_1 = CT_transform(1)
    # ms_l, ms_s = CT_4.contourlet_decompose(ms)
    # pan_l1, pan_s1 = CT_1.contourlet_decompose(pan)
    # pan_l2, pan_s2 = CT_1.contourlet_decompose(pan_l1)
    # pan_l3, pan_s3 = CT_1.contourlet_decompose(pan_l2)
    net = CT_10_10().to(device)
    y, l = net(ms, pan)
    # print(y.shape, l)


def test_mi():
    x = torch.zeros([20, 512, 1, 1])
    y = torch.ones([20, 512, 1, 1])
    print(mutual_information(x, x),
          mutual_information(y, y),
          mutual_information(x, y),
          mutual_information(y, x))


if __name__ == '__main__':
    test_net()