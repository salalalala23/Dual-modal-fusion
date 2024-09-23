import torch
import torch.nn as nn
import torch.nn.functional as F
import model.CT as CT
import time
from .resnet import BasicBlk

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
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


class H_ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(H_ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
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
        self.input_CA = ChannelAttention(input_dim)
        self.input_SA = SpatialAttention()

    def forward(self, input_tensor, cur_state):
        z_t = torch.sigmoid(self.conv_xz(input_tensor) + self.conv_hz(cur_state))
        r_t = torch.sigmoid(self.conv_xr(input_tensor) + self.conv_hr(cur_state))
        h_hat_t = torch.tanh(self.conv_xh(input_tensor) + self.conv_hh(torch.mul(r_t, cur_state)))
        h_t = torch.mul((1 - z_t), cur_state) + torch.mul(z_t, h_hat_t)
        out = self.conv_out(h_t)
        return h_t, out


# torch.size([batch_size, channel, width, height])
class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=max(inplanes // ratio, 1), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=max(inplanes // ratio, 1), out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out


# torch.size([batch_size, 1, width, height])
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


class CGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(CGRU, self).__init__()
        self.cell1 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell2 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell3 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell4 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell5 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)

    def forward(self, input_tensor, state=()):
        init_state = self.cell1.init_hidden(input_tensor)
        cur_state1, out = self.cell1(input_tensor, init_state)
        cur_state2, out = self.cell2(out+input_tensor, cur_state1)
        cur_state3, out = self.cell3(out+input_tensor, cur_state2)
        cur_state4, out = self.cell4(out+input_tensor, cur_state3)
        cur_state5, out = self.cell5(out+input_tensor, cur_state4)
        return out, (cur_state1, cur_state2, cur_state3, cur_state4, cur_state5)


class H_CGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(H_CGRU, self).__init__()
        self.cell1 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell2 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell3 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell4 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell5 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
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
        # else:
        #     init_state = self.cell1.init_hidden(input_tensor)
        #     cur_state1, out = self.cell1(input_tensor, torch.mul(init_state, self.unsample(state[0])))
        #     cur_state2, out = self.cell2(out+input_tensor, torch.mul(cur_state1, self.h_CA1(self.unsample(state[1]))))
        #     cur_state3, out = self.cell3(out+input_tensor, torch.mul(cur_state2, self.h_CA1(self.unsample(state[2]))))
        #     cur_state4, out = self.cell4(out+input_tensor, torch.mul(cur_state3, self.h_CA1(self.unsample(state[3]))))
        #     cur_state5, out = self.cell5(out+input_tensor, torch.mul(cur_state4, self.h_CA1(self.unsample(state[4]))))
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
        self.cell1 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell2 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.cell3 = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
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


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in inputbn
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class CT_transform(nn.Module):
    def __init__(self, channel, name='thanh'):
        super(CT_transform, self).__init__()
        self.name = name
        self.c = channel
        h = torch.tensor([.037828455506995, -.023849465019380,
                          -.11062440441842, .37740285561265, .85269867900940,
                          .37740285561265, -.11062440441842, -.023849465019380,
                          .037828455506995])
        h = torch.unsqueeze(h, 1)
        h = h * h.T.unsqueeze(0).expand(self.c, 1, 9, 9)
        g = torch.tensor([-.064538882628938, -.040689417609558,
                          .41809227322221, .78848561640566,
                          .41809227322221, -.040689417609558,
                          -.064538882628938])
        g = torch.unsqueeze(g, 1)
        g = g * g.T.unsqueeze(0).expand(self.c, 1, 7, 7)
        g0 = - torch.tensor([[0, -1, 0],
                             [-1, -4, -1],
                             [0, -1, 0]]) / 4.0
        g0 = g0.expand(self.c, 1, 3, 3)
        g1 = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, -1, 0, 0],
                           [0, 0, 0, -2, -4, -2, 0],
                           [0, 0, -1, -4, 28, -4, -1],
                           [0, 0, 0, -2, -4, -2, 0],
                           [0, 0, 0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]]) / 32.0
        g1 = g1.expand(self.c, 1, 7, 7)
        h0 = torch.tensor([[0, 0, -1, 0, 0],
                           [0, -2, 4, -2, 0],
                           [-1, 4, 28, 4, -1],
                           [0, -2, 4, -2, 0],
                           [0, 0, -1, 0, 0]]) / 32.0
        h0 = h0.expand(self.c, 1, 5, 5)
        h1 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, -1, 0, 0, 0],
                           [-1, 4, -1, 0, 0],
                           [0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0]]) / 4.0
        h1 = h1.expand(self.c, 1, 5, 5)
        self.h = h.to(device)
        self.g = g.to(device)
        self.g0 = g0.to(device)
        self.g1 = g1.to(device)
        self.h0 = h0.to(device)
        self.h1 = h1.to(device)

    def lp_dec(self, img):
        # h, g = h.to(device), g.to(device)
        height, w = img.shape[2], img.shape[3]
        pad_h = int(self.h.shape[2] / 2)
        padding_per1 = torch.nn.ReflectionPad2d((pad_h, pad_h, pad_h, pad_h))
        low = F.conv2d(padding_per1(img), self.h, groups=self.c)
        low = low[:, :, ::2, ::2]
        high = torch.zeros(img.shape).to(device)
        high[:, :, ::2, ::2] = low
        pad_g = int(self.g.shape[2] / 2)
        padding_per2 = torch.nn.ReflectionPad2d((pad_g, pad_g, pad_g, pad_g))
        high = F.conv2d(padding_per2(high), self.g, groups=self.c)
        high = img - high
        return low, high

    def q_sampling(self, img, q_mode='q0', op_mode='down'):
        # img = img.to(device)
        h, w = img.shape[2], img.shape[3]
        pad = torch.nn.ReflectionPad2d((w // 2, w // 2, h // 2, h // 2))
        img = pad(img)

        if q_mode == 'q0' and op_mode == 'down':
            q = torch.tensor([[1, -1, 0], [1, 1, 0]])
        elif q_mode == 'q1' and op_mode == 'down':
            q = torch.tensor([[1, 1, 0], [-1, 1, 0]])
        elif q_mode == 'q0' and op_mode == 'up':
            q = torch.tensor([[0.5, 0.5, 0], [-0.5, 0.5, 0]])
        elif q_mode == 'q1' and op_mode == 'up':
            q = torch.tensor([[0.5, -0.5, 0], [0.5, 0.5, 0]])
        else:
            raise NotImplementedError("Not available q type")

        q = q[None, ...].type(torch.FloatTensor).repeat(img.shape[0], 1, 1)
        grid = F.affine_grid(q, img.size(), align_corners=True).type(torch.FloatTensor).to(device)
        img = F.grid_sample(img, grid, align_corners=True)

        h, w = img.shape[2], img.shape[3]
        img = img[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4]
        return img

    def dfb_dec(self, img, index='', name=None):
        h, w = img.shape[2], img.shape[3]
        if name == 'haar':
            padding0 = (0, 1)
            padding1 = (0, 1)
        else:
            pass

        padding_per_2 = torch.nn.ReflectionPad2d((2, 2, 2, 2))

        y0 = self.q_sampling(F.conv2d(padding_per_2(img), self.h0, groups=self.c), q_mode='q0', op_mode='down')
        y1 = self.q_sampling(F.conv2d(padding_per_2(img), self.h1, groups=self.c), q_mode='q0', op_mode='down')

        y00 = self.q_sampling(F.conv2d(padding_per_2(y0), self.h0, groups=self.c), q_mode='q1', op_mode='down')
        y01 = self.q_sampling(F.conv2d(padding_per_2(y0), self.h1, groups=self.c), q_mode='q1', op_mode='down')
        y10 = self.q_sampling(F.conv2d(padding_per_2(y1), self.h0, groups=self.c), q_mode='q1', op_mode='down')
        y11 = self.q_sampling(F.conv2d(padding_per_2(y1), self.h1, groups=self.c), q_mode='q1', op_mode='down')
        return torch.cat((y00, y01, y10, y11), dim=1)[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4]

    def contourlet_decompose(self, img, index='', name='thanh'):
        # channel = img.shape[1]
        # 9-7 filters
        # h, g = filters.lp_filters(channel)
        # Laplacian Pyramid decompose
        time1 = time.time()
        low_band, high = self.lp_dec(img)
        # save_image(low_band, 'test_image/low_band'+index+'.png')
        # save_image(high, 'test_image/high_band'+index+'.png')
        # DFB filters
        # h0, h1 = self.dfb_filters(channel, mode='d', name=name)
        # DFB decompose
        sub_bands = self.dfb_dec(high, name=name)
        return low_band, sub_bands


class CT_LSTM_1(nn.Module):  # 用GRU搭建网络结构，以及降噪模块
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_1, self).__init__()
        self.in_planes = 64
        self.conv_m1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.conv_m2 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p2 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        # self.P_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_4 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_5 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_6 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_7 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        # self.P_8 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #                          nn.BatchNorm2d(1),
        #                          nn.ReLU(inplace=True))
        self.ups_1 = nn.Conv2d(20, 128, (3, 3), (1, 1), padding=1,bias=False)
        self.upp_1 = nn.Conv2d(5, 128, (3, 3), (1, 1), padding=1, bias=False)
        self.ups_2 = nn.Conv2d(20, 128, (3, 3), (1, 1), padding=1, bias=False)
        self.upp_2 = nn.Conv2d(5, 128, (3, 3), (1, 1), padding=1, bias=False)
        self.ups_3 = nn.Conv2d(20, 128, (3, 3), (1, 1), padding=1, bias=False)
        self.upp_3 = nn.Conv2d(5, 128, (3, 3), (1, 1), padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.GRU_s3 = ConvGRUCell(20, 20, (3, 3), False)
        self.GRU_l3 = ConvGRUCell(5, 5, (3, 3), False)
        self.GRU_s2 = ConvGRUCell(20, 20, (3, 3), False)
        self.GRU_l2 = ConvGRUCell(5, 5, (3, 3), False)
        self.GRU_s1 = ConvGRUCell(20, 20, (3, 3), False)
        self.GRU_l1 = ConvGRUCell(5, 5, (3, 3), False)
        self.conv_filter = nn.Conv2d(128, 32, (1, 1), (1, 1))
        self.conv64 = nn.Conv2d(32, 64, (1, 1), (1, 1))
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def denoise1(self, ms, pan):
    #     # torch.Size([20, 64, 16, 16]) torch.Size([20, 64, 16, 16])
    #     # noise reduction module
    #     f_ms, f_pan = einops.rearrange(ms, 'b c h w -> b c (h w)'), \
    #                   einops.rearrange(pan, 'b c h w -> b c (h w)')
    #     # torch.Size([20, 64, 256]) torch.Size([20, 64, 256])
    #     f_ms, f_pan = torch.unsqueeze(f_ms, dim=1), torch.unsqueeze(f_pan, dim=1)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #
    #     f_ms_main, f_ms_co = self.P_1(f_ms), self.P_2(f_ms)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #     f_pan_main, f_pan_co = self.P_3(f_pan), self.P_4(f_pan)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #     L_c1 = torch.mul(f_ms_main, f_ms_co)  # torch.Size([20, 1, 64, 256])
    #     L_c2 = torch.mul(f_pan_main, f_pan_co)  # torch.Size([20, 1, 64, 256])
    #     sim = torch.sigmoid(torch.cosine_similarity(f_pan_main, f_ms_main, dim=1))  # torch.Size([20, 64, 256])
    #     sim = torch.unsqueeze(sim, dim=1)  # torch.Size([20, 1, 64, 256])
    #
    #     f_ms_mains = torch.mul(sim, f_ms_main)  # torch.Size([20, 1, 64, 256])
    #     f_pan_mains = torch.mul(1 - sim, f_pan_main)  # torch.Size([20, 1, 64, 256])
    #     f_ms_s = torch.mul(f_ms_mains, f_ms_co)  # torch.Size([20, 1, 64, 256])
    #     f_pan_s = torch.mul(f_pan_mains, f_pan_co)  # torch.Size([20, 1, 64, 256])
    #     f_ms_s = einops.rearrange(f_ms_s, 'b a c (d w) ->b (a c) d w', d=ms.shape[2], w=ms.shape[2])  # torch.Size([20, 64, 16, 16])
    #     f_pan_s = einops.rearrange(f_pan_s, 'b a c (d w) ->b (a c) d w', d=ms.shape[2], w=ms.shape[2])  # torch.Size([20, 64, 16, 16])
    #     loss = nn.MSELoss()
    #     L = loss(L_c1, L_c2)
    #     return torch.cat((f_ms_s, f_pan_s), dim=1), L

    # def denoise2(self, ms, pan):
    #     # torch.Size([20, 64, 16, 16]) torch.Size([20, 64, 16, 16])
    #     # noise reduction module
    #     f_ms, f_pan = einops.rearrange(ms, 'b c h w -> b c (h w)'), \
    #                   einops.rearrange(pan, 'b c h w -> b c (h w)')
    #     # torch.Size([20, 64, 256]) torch.Size([20, 64, 256])
    #     f_ms, f_pan = torch.unsqueeze(f_ms, dim=1), torch.unsqueeze(f_pan, dim=1)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #
    #     f_ms_main, f_ms_co = self.P_5(f_ms), self.P_6(f_ms)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #     f_pan_main, f_pan_co = self.P_7(f_pan), self.P_8(f_pan)
    #     # torch.Size([20, 1, 64, 256]) torch.Size([20, 1, 64, 256])
    #     L_c1 = torch.mul(f_ms_main, f_ms_co)  # torch.Size([20, 1, 64, 256])
    #     L_c2 = torch.mul(f_pan_main, f_pan_co)  # torch.Size([20, 1, 64, 256])
    #     sim = torch.sigmoid(torch.cosine_similarity(f_pan_main, f_ms_main, dim=1))  # torch.Size([20, 64, 256])
    #     sim = torch.unsqueeze(sim, dim=1)  # torch.Size([20, 1, 64, 256])
    #
    #     f_ms_mains = torch.mul(sim, f_ms_main)  # torch.Size([20, 1, 64, 256])
    #     f_pan_mains = torch.mul(1 - sim, f_pan_main)  # torch.Size([20, 1, 64, 256])
    #     f_ms_s = torch.mul(f_ms_mains, f_ms_co)  # torch.Size([20, 1, 64, 256])
    #     f_pan_s = torch.mul(f_pan_mains, f_pan_co)  # torch.Size([20, 1, 64, 256])
    #     f_ms_s = einops.rearrange(f_ms_s, 'b a c (d w) ->b (a c) d w', d=ms.shape[2], w=ms.shape[2])  # torch.Size([20, 64, 16, 16])
    #     f_pan_s = einops.rearrange(f_pan_s, 'b a c (d w) ->b (a c) d w', d=ms.shape[2], w=ms.shape[2])  # torch.Size([20, 64, 16, 16])
    #     loss = nn.MSELoss()
    #     L = loss(L_c1, L_c2)
    #     return torch.cat((f_ms_s, f_pan_s), dim=1), L

    def forward(self, ms, pan):
        epoch = 3
        # CT module

        ms_l1, ms_s1 = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        pan_l1, pan_s1 = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])

        ms_l2, ms_s2 = CT.contourlet_decompose(ms_l1)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
        pan_l2, pan_s2 = CT.contourlet_decompose(pan_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])

        ms_l3, ms_s3 = CT.contourlet_decompose(ms_l2)  # torch.Size([20, 4, 2, 2]) torch.Size([20, 16, 2, 2])
        pan_l3, pan_s3 = CT.contourlet_decompose(pan_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])

        # f_s3, l1 = self.denoise1(self.conv_m1(ms_s3),
        #                          F.interpolate(self.conv_p1(pan_s3), size=(ms_s3.shape[2], ms_s3.shape[2])))
        # f_l3, l2 = self.denoise2(self.conv_m2(ms_l3),
        #                          F.interpolate(self.conv_p2(pan_l3), size=(ms_l3.shape[2], ms_l3.shape[2])))
        #
        # f_s2, l3 = self.denoise1(self.conv_m1(ms_s2),
        #                          F.interpolate(self.conv_p1(pan_s2), size=(ms_s2.shape[2], ms_s2.shape[2])))
        # f_l2, l4 = self.denoise2(self.conv_m2(ms_l2),
        #                          F.interpolate(self.conv_p2(pan_l2), size=(ms_l2.shape[2], ms_l2.shape[2])))
        #
        # f_s1, l5 = self.denoise1(self.conv_m1(ms_s1),
        #                          F.interpolate(self.conv_p1(pan_s1), size=(ms_s1.shape[2], ms_s1.shape[2])))
        # f_l1, l6 = self.denoise2(self.conv_m2(ms_l1),
        #                          F.interpolate(self.conv_p2(pan_l1), size=(ms_l1.shape[2], ms_l1.shape[2])))
        # print(ms_s3.shape, pan_s3.shape, f_s3.shape)
        f_s3 = torch.concat((ms_s3, F.interpolate(pan_s3, size=(ms_s3.shape[2], ms_s3.shape[3]))), dim=1)
        f_l3 = torch.concat((ms_l3, F.interpolate(pan_l3, size=(ms_l3.shape[2], ms_l3.shape[3]))), dim=1)
        f_s2 = torch.concat((ms_s2, F.interpolate(pan_s2, size=(ms_s2.shape[2], ms_s2.shape[3]))), dim=1)
        f_l2 = torch.concat((ms_l2, F.interpolate(pan_l2, size=(ms_l2.shape[2], ms_l2.shape[3]))), dim=1)
        f_s1 = torch.concat((ms_s1, F.interpolate(pan_s1, size=(ms_s1.shape[2], ms_s1.shape[3]))), dim=1)
        f_l1 = torch.concat((ms_l1, F.interpolate(pan_l1, size=(ms_l1.shape[2], ms_l1.shape[3]))), dim=1)

        h_s3 = self.GRU_s3(f_s3, self.GRU_s3.init_hidden(f_s3))
        h_l3 = self.GRU_l3(f_l3, self.GRU_l3.init_hidden(f_l3))
        r_3 = CT.contourlet_recompose(self.conv_filter(h_l3), h_s3)

        h_s2 = self.GRU_s2(f_s2, self.unsample(h_s3))
        h_l2 = self.GRU_l2(f_l2, self.unsample(h_l3))
        r_2 = CT.contourlet_recompose(self.conv_filter(h_l2), h_s2)

        h_s1 = self.GRU_s2(f_s1, self.unsample(h_s2))
        h_l1 = self.GRU_l2(f_l1, self.unsample(h_l2))
        r_1 = CT.contourlet_recompose(self.conv_filter(h_l1), h_s1)

        out = r_1 + self.unsample(r_2 + self.unsample(r_3))
        out = self.conv64(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, 1#, l1+l2+l3+l4+l5+l6

        # out = r1 + self.
        # out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        # out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        # out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        # out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        # out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        # out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        # out = self.linear(out)  # torch.Size([20, 12])
        # return out, [f_ms, L_c1, f_pan, L_c2]


class CT_LSTM_2(nn.Module):  # 去掉denoise模块
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_2, self).__init__()
        self.in_planes = 64
        self.conv_m1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.conv_m2 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p2 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.CGRU_s3 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l3 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s2 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l2 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s1 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l1 = CGRU(5, 5, (3, 3), False)
        self.conv_filter = nn.Conv2d(128, 32, (1, 1), (1, 1))
        self.conv64 = nn.Conv2d(5, 64, (1, 1), (1, 1))
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, ms, pan):
        epoch = 3
        # CT module

        ms_l1, ms_s1 = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        pan_l1, pan_s1 = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])

        ms_l2, ms_s2 = CT.contourlet_decompose(ms_l1)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
        pan_l2, pan_s2 = CT.contourlet_decompose(pan_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])

        # ms_l3, ms_s3 = CT.contourlet_decompose(ms_l2)  # torch.Size([20, 4, 2, 2]) torch.Size([20, 16, 2, 2])
        # pan_l3, pan_s3 = CT.contourlet_decompose(pan_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        #
        # f_s3 = torch.concat((ms_s3, F.interpolate(pan_s3, size=(ms_s3.shape[2], ms_s3.shape[3]))), dim=1)
        # f_l3 = torch.concat((ms_l3, F.interpolate(pan_l3, size=(ms_l3.shape[2], ms_l3.shape[3]))), dim=1)
        f_s2 = torch.concat((ms_s2, F.interpolate(pan_s2, size=(ms_s2.shape[2], ms_s2.shape[3]))), dim=1)
        f_l2 = torch.concat((ms_l2, F.interpolate(pan_l2, size=(ms_l2.shape[2], ms_l2.shape[3]))), dim=1)

        f_s1 = torch.concat((ms_s1, F.interpolate(pan_s1, size=(ms_s1.shape[2], ms_s1.shape[3]))), dim=1)
        f_l1 = torch.concat((ms_l1, F.interpolate(pan_l1, size=(ms_l1.shape[2], ms_l1.shape[3]))), dim=1)

        # h_s3 = self.GRU_s3(f_s3, self.GRU_s3.init_hidden(f_s3))
        # h_l3 = self.GRU_l3(f_l3, self.GRU_l3.init_hidden(f_l3))
        # r_3 = CT.contourlet_recompose(self.conv_filter(h_l3), h_s3)

        h_s2, state_s2 = self.CGRU_s2(f_s2)
        h_l2, state_l2 = self.CGRU_l2(f_l2)
        r_2 = CT.contourlet_recompose(h_l2, h_s2)

        h_s1, state_s1 = self.CGRU_s1(f_s1)
        h_l1, state_l1 = self.CGRU_l1(f_l1)
        r_1 = CT.contourlet_recompose(h_l1, h_s1)

        out = r_1 + self.unsample(r_2)
        out = self.conv64(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, 1


class CT_LSTM_3(nn.Module):  # 去掉GRU,单纯测试稍微处理的CT变换（重写了gru模块，不需要了）
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_3, self).__init__()
        self.in_planes = 64
        self.conv_m1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.conv_m2 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p2 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.CGRU_s3 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l3 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s2 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l2 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s1 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l1 = CGRU(5, 5, (3, 3), False)
        self.conv_filter = nn.Conv2d(128, 32, (1, 1), (1, 1))
        self.conv64 = nn.Conv2d(5, 64, (1, 1), (1, 1))
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, ms, pan):
        epoch = 3
        # CT module

        ms_l1, ms_s1 = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        pan_l1, pan_s1 = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])

        ms_l2, ms_s2 = CT.contourlet_decompose(ms_l1)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
        pan_l2, pan_s2 = CT.contourlet_decompose(pan_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])

        # ms_l3, ms_s3 = CT.contourlet_decompose(ms_l2)  # torch.Size([20, 4, 2, 2]) torch.Size([20, 16, 2, 2])
        # pan_l3, pan_s3 = CT.contourlet_decompose(pan_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        #
        # f_s3 = torch.concat((ms_s3, F.interpolate(pan_s3, size=(ms_s3.shape[2], ms_s3.shape[3]))), dim=1)
        # f_l3 = torch.concat((ms_l3, F.interpolate(pan_l3, size=(ms_l3.shape[2], ms_l3.shape[3]))), dim=1)
        f_s2 = torch.concat((ms_s2, F.interpolate(pan_s2, size=(ms_s2.shape[2], ms_s2.shape[3]))), dim=1)
        f_l2 = torch.concat((ms_l2, F.interpolate(pan_l2, size=(ms_l2.shape[2], ms_l2.shape[3]))), dim=1)

        f_s1 = torch.concat((ms_s1, F.interpolate(pan_s1, size=(ms_s1.shape[2], ms_s1.shape[3]))), dim=1)
        f_l1 = torch.concat((ms_l1, F.interpolate(pan_l1, size=(ms_l1.shape[2], ms_l1.shape[3]))), dim=1)

        # h_s3 = self.GRU_s3(f_s3, self.GRU_s3.init_hidden(f_s3))
        # h_l3 = self.GRU_l3(f_l3, self.GRU_l3.init_hidden(f_l3))
        # r_3 = CT.contourlet_recompose(self.conv_filter(h_l3), h_s3)
        r_2 = CT.contourlet_recompose(f_l2, f_s2)
        r_1 = CT.contourlet_recompose(f_l1, f_s1)

        out = r_1 + self.unsample(r_2)
        out = self.conv64(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, 1


class CT_LSTM_4(nn.Module):  # 在3的基础上，对高频信息通过GRU稍微处理，查看结果（重写了gru模块，不需要了）
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_4, self).__init__()
        self.in_planes = 64
        self.conv_m1 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.conv_m2 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv_p2 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.CGRU_s3 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l3 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s2 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l2 = CGRU(5, 5, (3, 3), False)
        self.CGRU_s1 = CGRU(20, 20, (3, 3), False)
        self.CGRU_l1 = CGRU(5, 5, (3, 3), False)
        self.conv_filter = nn.Conv2d(128, 32, (1, 1), (1, 1))
        self.conv64 = nn.Conv2d(5, 64, (1, 1), (1, 1))
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, ms, pan):
        epoch = 3
        # CT module

        ms_l1, ms_s1 = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        pan_l1, pan_s1 = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])

        ms_l2, ms_s2 = CT.contourlet_decompose(ms_l1)  # torch.Size([20, 4, 4, 4]) torch.Size([20, 16, 4, 4])
        pan_l2, pan_s2 = CT.contourlet_decompose(pan_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])

        # ms_l3, ms_s3 = CT.contourlet_decompose(ms_l2)  # torch.Size([20, 4, 2, 2]) torch.Size([20, 16, 2, 2])
        # pan_l3, pan_s3 = CT.contourlet_decompose(pan_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        #
        # f_s3 = torch.concat((ms_s3, F.interpolate(pan_s3, size=(ms_s3.shape[2], ms_s3.shape[3]))), dim=1)
        # f_l3 = torch.concat((ms_l3, F.interpolate(pan_l3, size=(ms_l3.shape[2], ms_l3.shape[3]))), dim=1)
        f_s2 = torch.concat((ms_s2, F.interpolate(pan_s2, size=(ms_s2.shape[2], ms_s2.shape[3]))), dim=1)  # torch.size([20, 20, 4, 4])
        f_l2 = torch.concat((ms_l2, F.interpolate(pan_l2, size=(ms_l2.shape[2], ms_l2.shape[3]))), dim=1)  # torch.size([20, 5, 4, 4])

        f_s1 = torch.concat((ms_s1, F.interpolate(pan_s1, size=(ms_s1.shape[2], ms_s1.shape[3]))), dim=1)  # torch.size([20, 20, 8, 8])
        f_l1 = torch.concat((ms_l1, F.interpolate(pan_l1, size=(ms_l1.shape[2], ms_l1.shape[3]))), dim=1)  # torch.size([20, 5, 8, 8])

        # h_s3 = self.GRU_s3(f_s3, self.GRU_s3.init_hidden(f_s3))
        # h_l3 = self.GRU_l3(f_l3, self.GRU_l3.init_hidden(f_l3))
        # r_3 = CT.contourlet_recompose(self.conv_filter(h_l3), h_s3)

        r_2 = CT.contourlet_recompose(f_l2, f_s2)
        r_1 = CT.contourlet_recompose(f_l1, f_s1)

        out = r_1 + self.unsample(r_2)
        out = self.conv64(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, 1


class CT_LSTM_5(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_5, self).__init__()
        self.in_planes = 64

        self.c_p_l = nn.Conv2d(1, 16, (1, 1), (1, 1), bias=False)
        self.c_p_s = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_l = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_s = nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)

        self.CGRU_p_s3 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_s2 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_s1 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_l3 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_l2 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_l1 = H_CGRU(16, 20, (3, 3), False)

        self.CGRU_m_s1 = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_m_l1 = H_CGRU(16, 20, (3, 3), False)
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        # CT module
        m_l1, m_s1 = CT.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        m_l1, m_s1 = self.c_m_l(m_l1), self.c_m_s(m_s1)  # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 8, 8])

        p_l1, p_s1 = CT.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])
        p_l2, p_s2 = CT.contourlet_decompose(p_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])
        p_l3, p_s3 = CT.contourlet_decompose(p_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        p_l1, p_l2, p_l3 = self.c_p_l(p_l1), self.c_p_l(p_l2), self.c_p_l(p_l3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])
        p_s1, p_s2, p_s3 = self.c_p_s(p_s1), self.c_p_s(p_s2), self.c_p_s(p_s3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])

        p_out_s3, p_state_s3 = self.CGRU_p_s3(p_s3)  # torch.Size([20, 16, 8, 8])
        p_out_s2, p_state_s2 = self.CGRU_p_s2(p_s2, p_state_s3)  # torch.Size([20, 16, 16, 16])
        p_out_s1, p_state_s1 = self.CGRU_p_s1(p_s1, p_state_s2)  # torch.Size([20, 16, 32, 32])

        p_out_l3, p_state_l3 = self.CGRU_p_l3(p_l3)  # torch.size([20, 16, 8, 8])
        p_out_l2, p_state_l2 = self.CGRU_p_l2(p_l2, p_state_l3)  # torch.size([20, 16, 16, 16])
        p_out_l1, p_state_l1 = self.CGRU_p_l1(p_l1, p_state_l2)  # torch.size([20, 16, 32, 32])

        m_out_l, m_state_l = self.CGRU_m_l1(m_l1)  # torch.Size([20, 16, 8, 8])
        m_out_s, m_state_s = self.CGRU_m_s1(m_s1)  # torch.Size([20, 16, 8, 8])

        out = m_out_l
        # r_2 = CT.contourlet_recompose(h_l2, h_s2)
        #
        # h_s1, state_s1 = self.CGRU_s1(f_s1)
        # h_l1, state_l1 = self.CGRU_l1(f_l1)
        # r_1 = CT.contourlet_recompose(h_l1, h_s1)
        #
        # out = r_1 + self.unsample(r_2)
        # print(out.shape)
        # out = self.conv64(out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out, 1


class CT_LSTM_6(nn.Module):  # 添加新修改的多层RNN
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_6, self).__init__()
        self.in_planes = 64

        self.c_p_l = nn.Conv2d(1, 16, (1, 1), (1, 1), bias=False)
        self.c_p_s = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_l = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_s = nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)

        self.CGRU_p_s = DH_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_l = DH_CGRU(16, 20, (3, 3), False)

        self.CGRU_m_s = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_m_l = H_CGRU(16, 20, (3, 3), False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.conv_downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

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
        # CT module
        m_l1, m_s1 = self.ct_m.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        m_l1, m_s1 = self.c_m_l(m_l1), self.c_m_s(m_s1)  # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 8, 8])

        p_l1, p_s1 = self.ct_p.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])
        p_l2, p_s2 = self.ct_p.contourlet_decompose(p_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])
        p_l3, p_s3 = self.ct_p.contourlet_decompose(p_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        p_l1, p_l2, p_l3 = self.c_p_l(p_l1), self.c_p_l(p_l2), self.c_p_l(p_l3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])
        p_s1, p_s2, p_s3 = self.c_p_s(p_s1), self.c_p_s(p_s2), self.c_p_s(p_s3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])

        p_out_s, p_state_s = self.CGRU_p_s((p_s1, p_s2, p_s3))
        # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 32, 32])
        p_out_l, p_state_l = self.CGRU_p_l((p_l1, p_l2, p_l3))
        # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 32, 32])

        m_out_s, m_state_s = self.CGRU_m_l(m_s1, p_state_s[0])  # torch.Size([20, 16, 8, 8])
        m_out_l, m_state_l = self.CGRU_m_s(m_l1, p_state_l[0])  # torch.Size([20, 16, 8, 8])

        # scale1_s = torch.concat(self.denoise(p_out_s[0], m_out_s), dim=1)  # torch.Size([20, 32, 8, 8])
        # scale2_s = torch.concat(self.denoise(p_out_s[1], self.upsample2(m_out_s)), dim=1)
        # torch.Size([20, 32, 16, 16])
        # scale3_s = torch.concat(self.denoise(p_out_s[2], self.upsample4(m_out_s)), dim=1)
        # torch.Size([20, 32, 32, 32])

        # scale1_l = torch.concat(self.denoise(p_out_l[0], m_out_l), dim=1)
        # # torch.Size([20, 32, 8, 8])
        # scale2_l = torch.concat(self.denoise(p_out_l[1], self.upsample2(m_out_l)), dim=1)
        # # torch.Size([20, 32, 16, 16])
        # scale3_l = torch.concat(self.denoise(p_out_l[2], self.upsample4(m_out_l)), dim=1)
        # # torch.Size([20, 32, 32, 32])

        scale1_s = torch.concat((p_out_s[0], m_out_s), dim=1)  # torch.Size([20, 32, 8, 8])
        scale2_s = torch.concat((p_out_s[1], self.upsample2(m_out_s)),
                                dim=1)  # torch.Size([20, 32, 16, 16])
        scale3_s = torch.concat((p_out_s[2], self.upsample4(m_out_s)),
                                dim=1)  # torch.Size([20, 32, 32, 32])

        scale1_l = torch.concat((p_out_l[0], m_out_l), dim=1)
        # torch.Size([20, 32, 8, 8])
        scale2_l = torch.concat((p_out_l[1], self.upsample2(m_out_l)), dim=1)
        # torch.Size([20, 32, 16, 16])
        scale3_l = torch.concat((p_out_l[2], self.upsample4(m_out_l)), dim=1)
        # torch.Size([20, 32, 32, 32])

        out1 = torch.cat((scale1_s, scale1_l), dim=1)  # torch.Size([20, 64, 8, 8])
        out2 = torch.cat((scale2_s, scale2_l), dim=1)  # torch.Size([20, 64, 16, 16])
        out3 = torch.cat((scale3_s, scale3_l), dim=1)  # torch.Size([20, 64, 32, 32])

        out = self.conv_downsample1(out3) + out2 + self.upsample2(out1)
        out = self.layer1(out)  # torch.Size([20, 64, 16, 16])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])
        return out, 1


class CT_LSTM_7(nn.Module):  # 降重模块后置
    def __init__(self, block, num_blocks, num_classes):
        super(CT_LSTM_7, self).__init__()
        self.in_planes = 64

        self.c_p_l = nn.Conv2d(1, 16, (1, 1), (1, 1), bias=False)
        self.c_p_s = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_l = nn.Conv2d(4, 16, (1, 1), (1, 1), bias=False)
        self.c_m_s = nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)

        self.CGRU_p_s = DH_CGRU(16, 20, (3, 3), False)
        self.CGRU_p_l = DH_CGRU(16, 20, (3, 3), False)

        self.CGRU_m_s = H_CGRU(16, 20, (3, 3), False)
        self.CGRU_m_l = H_CGRU(16, 20, (3, 3), False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.downsample1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.downsample4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv_downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

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
        # CT module
        m_l1, m_s1 = self.ct_m.contourlet_decompose(ms)  # torch.Size([20, 4, 8, 8]) torch.Size([20, 16, 8, 8])
        m_l1, m_s1 = self.c_m_l(m_l1), self.c_m_s(m_s1)  # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 8, 8])

        p_l1, p_s1 = self.ct_p.contourlet_decompose(pan)  # torch.Size([20, 1, 32, 32]) torch.Size([20, 4, 32, 32])
        p_l2, p_s2 = self.ct_p.contourlet_decompose(p_l1)  # torch.Size([20, 1, 16, 16]) torch.Size([20, 4, 16, 16])
        p_l3, p_s3 = self.ct_p.contourlet_decompose(p_l2)  # torch.Size([20, 1, 8, 8]) torch.Size([20, 4, 8, 8])
        p_l1, p_l2, p_l3 = self.c_p_l(p_l1), self.c_p_l(p_l2), self.c_p_l(p_l3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])
        p_s1, p_s2, p_s3 = self.c_p_s(p_s1), self.c_p_s(p_s2), self.c_p_s(p_s3)
        # torch.Size([20, 16, 32, 32]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 8, 8])

        p_out_s, p_state_s = self.CGRU_p_s((p_s1, p_s2, p_s3))
        # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 32, 32])
        p_out_l, p_state_l = self.CGRU_p_l((p_l1, p_l2, p_l3))
        # torch.Size([20, 16, 8, 8]) torch.Size([20, 16, 16, 16]) torch.Size([20, 16, 32, 32])

        m_out_s, m_state_s = self.CGRU_m_l(m_s1, p_state_s[0])  # torch.Size([20, 16, 8, 8])
        m_out_l, m_state_l = self.CGRU_m_s(m_l1, p_state_l[0])  # torch.Size([20, 16, 8, 8])

        p_scale_s = self.downsample1(self.downsample2(p_out_s[2])+p_out_s[1]) + p_out_s[0]
        # torch.Size([20, 16, 8, 8])
        p_scale_l = self.downsample3(self.downsample4(p_out_l[2])+p_out_l[1]) + p_out_l[0]
        # torch.Size([20, 16, 8, 8])

        p = torch.concat((p_scale_s, p_scale_l), dim=1)  # torch.Size([20, 32, 8, 8])
        m = torch.concat((m_out_s, m_out_l), dim=1)  # torch.Size([20, 32, 8, 8])

        m1, p1 = self.denoise(m, p)
        MSE = nn.MSELoss()
        loss = MSE(m1, m) + MSE(p1, p)
        out = torch.concat((m1, p1), dim=1)  # torch.Size([20, 64, 8, 8])
        out = self.layer1(out)  # torch.Size([20, 64, 8, 8])
        out = self.layer2(out)  # torch.Size([20, 128, 8, 8])
        out = self.layer3(out)  # torch.Size([20, 256, 4, 4])
        out = self.layer4(out)  # torch.Size([20, 512, 2, 2])
        # out = F.avg_pool2d(out, 2)  # torch.Size([20, 512, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([20, 512])
        out = self.linear(out)  # torch.Size([20, 12])

        return out, loss


def CT_10_2():
    return CT_LSTM_2(BasicBlk, [1, 1, 1, 1])


def CT_10_6():
    return CT_LSTM_6(BasicBlk, [1, 1, 1, 1])


def CT_10_7():
    return CT_LSTM_7(BasicBlk, [1, 1, 1, 1])


def CT_18():
    return CT_LSTM_6(BasicBlk, [2, 2, 2, 2])


def CT_34():
    return CT_LSTM_6(BasicBlk, [3, 4, 6, 3])


def test_CT():
    img = torch.randn([20, 1, 64, 64]).to(device)
    CT = CT_transform(img.shape[1]).to(device)
    CT.contourlet_decompose(img)
    CT.contourlet_decompose(img)
    CT.contourlet_decompose(img)


def test_net():
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    # CT_4 = CT_transform(4)
    # CT_1 = CT_transform(1)
    # ms_l, ms_s = CT_4.contourlet_decompose(ms)
    # pan_l1, pan_s1 = CT_1.contourlet_decompose(pan)
    # pan_l2, pan_s2 = CT_1.contourlet_decompose(pan_l1)
    # pan_l3, pan_s3 = CT_1.contourlet_decompose(pan_l2)
    net = CT_10_7().to(device)
    y, l = net(ms, pan)
    # print(y.shape, l)


def test_CGRU():
    x = torch.randn([2, 4, 16, 16]).to(device)
    net = CGRU(4, 4, [3, 3], False).to(device)
    h, y = net(x)
    print(h.shape, len(y), y[0].shape)


def test_svd():
    # x = torch.randn([2, 16, 16, 16])
    # a, b, c = torch.linalg.svd(x)
    # d = torch.linalg.svdvals(x)
    # print(a.shape, b.shape, c.shape)
    # print(d.shape)
    ms = torch.randn([4, 4])
    pan = torch.randn([1, 4, 4, 4])
    m_u, m_sum, m_v = torch.linalg.svd(ms)
    f_ms = torch.matmul(torch.matmul(m_u, torch.diag_embed(m_sum)), m_v)
    print(ms, f_ms)
    print(torch.dist(ms, f_ms))
    p_u, p_sum, p_v = torch.linalg.svd(pan)
    sim = torch.sigmoid(torch.cosine_similarity(m_u, p_u, dim=1))
    sim = torch.unsqueeze(sim, dim=1)

    f_m_u = torch.mul(sim, m_u)
    f_p_u = torch.mul(1 - sim, p_u)
    f_m = torch.mul(f_m_u, m_v)
    f_p = torch.mul(f_p_u, p_v)
    return f_m, f_p


def test_DHGRU():
    net = DH_CGRU(input_dim=16, hidden_dim=20, kernel_size=(3, 3), bias=False)
    x = (torch.randn([20, 16, 8, 8]), torch.randn([20, 16, 16, 16]), torch.randn([20, 16, 32, 32]))
    print(x[0].shape[0])
    out = net(x)


if __name__ == '__main__':
    test_net()
