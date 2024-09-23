import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import *
from model.contourlet_torch import ContourDec, ContourRec
# from model.methods.contourlet import ContourDec, ContourRec
import pywt
import numpy as np
from model.INN import RevNetModel
# from model.panmamba import SingleMambaBlock


def WaveDec2d(x, mode='haar', l=1):
    coefs = pywt.wavedec2(x.cpu().detach(), mode, level=1)
    return coefs


def WaveRec2d(coefs, mode='haar', l=1):
    x = pywt.waverec2(coefs, mode)
    return torch.tensor(x)


def collision_probability_torch(tensor1, tensor2, epsilon):
    assert tensor1.shape == tensor2.shape, "Tensors must be of the same shape"

    # 计算两个张量之间的差异
    differences = torch.abs(tensor1 - tensor2)

    # 确定哪些元素是"碰撞的"（即差异小于epsilon）
    collisions = differences < epsilon

    # 计算碰撞的比例
    collision_count = torch.sum(collisions)
    total_elements = tensor1.numel()  # numel 返回张量中元素的总数

    # 计算单个碰撞的概率
    p_single_collision = collision_count.float() / total_elements

    # 应用生日悖论公式估计至少一次碰撞的概率
    n = total_elements
    p_at_least_one_collision = 1 - torch.exp(-n * (n - 1) / (2 * total_elements / p_single_collision))

    return p_at_least_one_collision

class Mambablock(nn.Module):
    def __init__(self, dim, inv=0):
        super(Mambablock, self).__init__()
        self.inv = inv
        self.block = Mamba(dim, bimamba_type=None)
        self.norm = nn.LayerNorm(dim)

    def forward(self, in_tensor):
        if len(in_tensor) == 2:
            x, res = in_tensor
            res = res + x
            x = self.block(self.norm(res))
            return (x, res) if self.inv == 0 else (res, x)
        else:
            return self.block(self.norm(in_tensor))


class mambaout(Mamba):
    def forward(self, hidden_states, inference_params=None):
        """
                hidden_states: (B, L, D)
                Returns: same shape as hidden_states
                """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        # y = selective_scan_fn(
        #     x,
        #     dt,
        #     A,
        #     B,
        #     C,
        #     self.D.float(),
        #     z=z,
        #     delta_bias=self.dt_proj.bias.float(),
        #     delta_softplus=True,
        #     return_last_state=ssm_state is not None,
        # )
        y = x
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=False):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm_1 = nn.BatchNorm2d(out_size)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm_2 = nn.BatchNorm2d(out_size)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        # resi = self.relu_1(self.conv_1(x))
        # out_1, out_2 = torch.chunk(resi, 2, dim=1)
        # resi = torch.cat([self.norm(out_1), out_2], dim=1)
        # resi = self.relu_2(self.conv_2(resi))
        # return x+resi
        res = self.norm_1(self.relu_1(self.conv_1(x)))
        x = x + res
        res = self.norm_2(self.relu_2(self.conv_2(x)))
        return x + res


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, stride=4, in_chans=32, embed_dim=32 * 32 * 32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Fre_cross(nn.Module):
    def __init__(self, dim=32, nlevs=3, mode='ct'):
        super().__init__()
        if mode == 'ct':
            self.decompose = ContourDec(nlevs)
            self.combine = ContourRec(nlevs)
        elif mode == 'wavelet':
            self.decompose = WaveDec2d
            self.combine = WaveRec2d
        else:
            raise AttributeError('change the right mode')

        self.s_inn = RevNetModel(dim*2**nlevs, 3, 2)
        self.l_inn = RevNetModel(dim, 3, 2)

    def forward(self, m, p):
        m_l, m_s = self.decompose(m)
        p_l, p_s = self.decompose(p)

        m_outs, p_outs = torch.chunk(self.s_inn(torch.cat([m_s, p_s], dim=1)), 2, 1)
        m_outl, p_outl = torch.chunk(self.l_inn(torch.cat([m_l, p_l], dim=1)), 2, 1)

        m_out = self.combine([m_outl, p_outs])
        p_out = self.combine([p_outl, m_outs])

        # m_out = self.combine([m_l, p_s])
        # p_out = self.combine([p_l, m_s])

        return m_out, p_out


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=4)
        num_classes = args['Categories_Number']
        in_chans = 32
        self.m_in = nn.Conv2d(4, in_chans, 3, 1, 1)
        self.p_in = nn.Conv2d(1, in_chans, 3, 1, 1)

        self.Fre_inn = Fre_cross(in_chans)

        self.m_encoder = nn.Sequential(
            *[HinResBlock(in_chans, in_chans) for i in range(3)]
        )
        self.p_encoder = nn.Sequential(
            *[HinResBlock(in_chans, in_chans) for i in range(3)]
        )
        # mamba_layer = 3
        # self.m_encoder = nn.Sequential(
        #     *[Mambablock(in_chans) for _ in range(mamba_layer)]
        # )
        # self.p_encoder = nn.Sequential(
        #     *[Mambablock(in_chans) for _ in range(mamba_layer)]
        # )

        self.m_part = PatchEmbed(1, 1, in_chans, in_chans)
        self.p_part = PatchEmbed(1, 1, in_chans, in_chans)

        mamba_layer = 8
        self.m_mamba = nn.Sequential(
            *[Mambablock(in_chans) for _ in range(mamba_layer)]
        )
        self.p_mamba = nn.Sequential(
            *[Mambablock(in_chans) for _ in range(mamba_layer)]
        )

        mamba_layer = 8
        self.inv_mamba = nn.Sequential(
            *[Mambablock(in_chans, 1) for _ in range(mamba_layer)]
        )

        self.inv_mamba2 = nn.Sequential(
            *[Mambablock(in_chans, 1) for _ in range(mamba_layer)]
        )

        # self.inn1 = nn.Sequential(
        #     *[RevNetModel(num_channels=in_chans, kernel=3, num_layers=2) for _ in range(1)]
        # )
        # self.inn2 = nn.Sequential(
        #     *[RevNetModel(num_channels=in_chans, kernel=3, num_layers=2) for _ in range(1)]
        # )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4096*32, in_chans)
        self.linear2 = nn.Linear(in_chans, num_classes)


    def forward(self, ms, pan):
        B, _, H, W = pan.shape
        m = self.upsample(ms)
        m_in = self.m_in(m)
        p_in = self.p_in(pan)

        m_out, p_out = self.Fre_inn(m_in, p_in)

        # CT module
        # m_l, m_s = ContourDec(3)(m_in)
        # p_l, p_s = ContourDec(3)(p_in)
        # m_out = ContourRec(3)([m_l, p_s])
        # p_out = ContourRec(3)([p_l, m_s])

        # Wavelet module
        # m_l, m_s = WaveDec2d(m_in)
        # p_l, p_s = WaveDec2d(p_in)
        # m_out = WaveRec2d([m_l, p_s]).to(ms.device)
        # p_out = WaveRec2d([p_l, m_s]).to(ms.device)

        # m_out = self.m_in(m)
        # p_out = self.p_in(pan)

        # m_out, p_out = torch.chunk(self.inn1(torch.cat([m_out, p_out], dim=1)), 2, 1)

        m_out = self.m_encoder(m_out)
        p_out = self.p_encoder(p_out)

        m_out = self.m_part(m_out)
        p_out = self.p_part(p_out)
        m_out, p_out = self.inv_mamba([m_out, p_out])

        residual_ms_f = 0
        residual_pan_f = 0
        m_out = self.m_mamba([m_out, residual_ms_f])[0]  # .transpose(1, 2).view(B, -1, H, W)
        p_out = self.p_mamba([p_out, residual_pan_f])[0]  # .transpose(1, 2).view(B, -1, H, W)

        out = m_out + p_out
        out = self.linear2(self.linear1(self.flatten(out)))

        # m_out, p_out = self.inv_mamba2([m_out, p_out])
        # m_out = self.m_mamba(self.m_part(m_out)).transpose(1, 2).view(B, -1, H, W)
        # p_out = self.p_mamba(self.p_part(p_out)).transpose(1, 2).view(B, -1, H, W)

        # m_out, p_out = torch.chunk(self.inn2(torch.cat([m_out, p_out], dim=1)), 2, 1)
        # m_out = self.m_out(m_out.transpose(1, 2).view(B, -1, H, W))
        # p_out = self.p_out(p_out.transpose(1, 2).view(B, -1, H, W))
        # out = self.m_out(m_out.transpose(1, 2).view(B, -1, H, W)) + self.p_out(p_out.transpose(1, 2).view(B, -1, H, W))
        # indicator(m_out, p_out)
        return out


class Best_Net(nn.Module):
    def __init__(self, args):
        super(Best_Net, self).__init__()
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=4)
        num_channels = args['num_channels']
        in_chans = 32
        self.m_in = nn.Conv2d(num_channels, in_chans, 3, 1, 1)
        self.p_in = nn.Conv2d(1, in_chans, 3, 1, 1)

        self.m_encoder = nn.Sequential(
            *[HinResBlock(in_chans, in_chans) for i in range(3)]
        )
        self.p_encoder = nn.Sequential(
            *[HinResBlock(in_chans, in_chans) for i in range(3)]
        )

        self.m_part = PatchEmbed(1, 1, in_chans, in_chans)
        self.p_part = PatchEmbed(1, 1, in_chans, in_chans)

        mamba_layer = 8
        self.m_mamba = nn.Sequential(
            *[Mambablock(in_chans) for _ in range(mamba_layer)]
        )
        self.p_mamba = nn.Sequential(
            *[Mambablock(in_chans) for _ in range(mamba_layer)]
        )

        mamba_layer = 8
        self.inv_mamba = nn.Sequential(
            *[Mambablock(in_chans, 1) for _ in range(mamba_layer)]
        )

        self.inv_mamba2 = nn.Sequential(
            *[Mambablock(in_chans, 1) for _ in range(mamba_layer)]
        )

        self.m_out = nn.Conv2d(in_chans, num_channels, 3, 1, 1)
        self.p_out = nn.Conv2d(in_chans, 1, 3, 1, 1)

    def forward(self, ms, pan):
        B, _, H, W = pan.shape
        m = self.upsample(ms)
        m_in = self.m_in(m)
        p_in = self.p_in(pan)

        m_l, m_s = ContourDec(2)(m_in)
        p_l, p_s = ContourDec(2)(p_in)
        m_out = ContourRec(2)([m_l, p_s])
        p_out = ContourRec(2)([p_l, m_s])

        m_out = self.m_encoder(m_out)
        p_out = self.p_encoder(p_out)

        m_out = self.m_part(m_out)
        p_out = self.p_part(p_out)
        m_out, p_out = self.inv_mamba([m_out, p_out])

        residual_ms_f = 0
        residual_pan_f = 0
        m_out = self.m_mamba([m_out, residual_ms_f])[0]
        p_out = self.p_mamba([p_out, residual_pan_f])[0]

        out = self.m_out(m_out.transpose(1, 2).view(B, -1, H, W)) + self.p_out(p_out.transpose(1, 2).view(B, -1, H, W))
        return out + m


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


def indicator(tensor1, tensor2):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    coll = collision_probability_torch(tensor1, tensor2, epsilon=1e-6)
    print(mse, coll)


def test_coll():
    m = torch.randn([20, 8, 32, 32])
    p = torch.randn([20, 8, 32, 32])
    mse = torch.mean((m - p) ** 2)
    coll = collision_probability_torch(m, p, epsilon=1e-6)
    print(mse, coll)


if __name__ == '__main__':
    test_net()
    # test_coll()
    # ms = torch.randn([20, 32, 32, 32]).to('cuda')
    # s, l = WaveDec2d(ms)
    # y = WaveRec2d([s, l])
    # print(y.shape)