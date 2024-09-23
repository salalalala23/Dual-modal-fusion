import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, Softmax
import copy
from einops import rearrange, repeat

patchsize = 16


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


"""
class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g, padding=kernel_size // 3,
                             stride=stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)
        self.out_channels = out_channels  # 添加l此行代码

    def forward(self, x):
        if x.size(1) != self.out_channels:  # 添加l此行代码
            raise ValueError("Input tensor channel number does not match with the specified in_channels.")
        return self.gwc(x) + self.pwc(x)"""


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        # 计算分组数，确保输入通道数能够被分组数整除
        groups_g = min(g, in_channels)  # 分组数不超过输入通道数
        groups_p = min(p, in_channels)  # 分组数不超过输入通道数

        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=4, padding=kernel_size // 3,
                             stride=stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim, bias=qkv_bias)
        #         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        #         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        #         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        #         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)

        encoded = self.encoder_norm(x)

        return encoded[:, 0]


class MFT(nn.Module):
    def __init__(self, FM=16, NC=4, NCLidar=1, Classes=15, HSIOnly=False):
        super(MFT, self).__init__()
        self.HSIOnly = HSIOnly
        """        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU()

        )"""
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), padding=(0, 1, 1), stride=1),  # 将 kernel_size 改为 (3, 3, 3)
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        """      self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
                    ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )"""
        self.conv6 = nn.Sequential(
            HetConv(4, FM * 4,
                    p=1,
                    g=(FM * 4) // 4 if 4 % FM == 0 else (FM * 4) // 8,
                    ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )
        #        input_channels = x1.size(1)  # 获取输入 x1 的通道数
        self.conv_adjust1 = nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0)
        self.conv_adjust2 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

        self.last_BandSize = NC // 2 // 2 // 2

        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidar, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.ca = TransformerEncoder(FM * 4)
        self.out3 = nn.Linear(FM * 4, Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, FM * 4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV_L)

    def forward(self, x1, x2):
        #       print("x1 shape:", x1.shape)  # 添加打印语句，打印输入张量 x1 的形状
        x1 = x1.reshape(x1.shape[0], 4, patchsize, patchsize)

        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0], -1, patchsize, patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, patchsize, patchsize)

        # 假设输入 x1 的通道数为 input_channels
        #        input_channels = x1.size(1)  # 获取输入 x1 的通道数
        #       print(input_channels)
        # 定义一个卷积层，将输入 x1 的通道数调整为合适的数值，以匹配 self.conv6 的输入通道数
        #        conv_adjust = nn.Conv2d(input_channels, 4, kernel_size=1, stride=1, padding=0)

        # 对输入 x1 进行通道数调整，使其与 self.conv6 的输入通道数相匹配
        x1 = self.conv_adjust1(x1)
        x2 = self.conv_adjust2(x2)

        x1 = self.conv6(x1)
        x2 = self.lidarConv(x2)
        x2 = x2.reshape(x2.shape[0], -1, patchsize ** 2)
        x2 = x2.transpose(-1, -2)

        wa_L = self.token_wA_L.expand(x1.shape[0], -1, -1)
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose
        A_L = A_L.softmax(dim=-1)

        wv_L = self.token_wV_L.expand(x2.shape[0], -1, -1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        x2 = torch.einsum('bij,bjk->bik', A_L, VV_L)

        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)

        wa = self.token_wA.expand(x1.shape[0], -1, -1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        wv = self.token_wV.expand(x1.shape[0], -1, -1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((x2, T), dim=1)  # [b, n+1, dim]

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings)
        x = x.reshape(x.shape[0], -1)
        out3 = self.out3(x)
        return out3


def Net(args):
    # 这里的第二个参数应该是整数值，表示层次块的数量，而不是 Block 类型
    return MFT(Classes=args['Categories_Number'])


# test()

def test_net():
    device = 'cuda'
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8,
        'device': 'cuda:0'
    }
    net = Net(cfg).to(device)
    y = net(ms, pan)
    print(y.shape)


if __name__ == '__main__':
    test_net()
