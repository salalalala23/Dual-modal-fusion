import torch
from torch import nn

from model.panform.common.modules import conv3x3, SwinModule
# from model.panform.base_model import Base_model
# from model.panform.builder import MODELS


class CrossSwinTransformer(nn.Module):
    def __init__(self, ms_chans=4, n_feats=64, n_heads=4, head_dim=16, win_size=4, num_classes=8,
                 n_blocks=2, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False):
        super().__init__()
        # self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=ms_chans, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]

        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(n_blocks):
                self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            ms_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                         downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                                         window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            pan_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                          downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                                          window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        self.HR_tail = nn.Sequential(
            conv3x3(n_feats * len(cat_feat), n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, ms_chans))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

        self.classifier = nn.Sequential(
            nn.Linear(n_feats * 2 * 4 * 4, n_feats),
            nn.GELU(), nn.Linear(n_feats, num_classes)
        )

    def forward(self, ms, pan):
        b = ms.shape[0]
        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms)

        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        for i in range(self.n_blocks):
            if 'pan' in self.cross_module:
                pan_cross_ms_feat = self.pan_cross_ms[i](last_pan_feat, last_ms_feat)
            if 'ms' in self.cross_module:
                ms_cross_pan_feat = self.ms_cross_pan[i](last_ms_feat, last_pan_feat)
            if 'pan' in self.cross_module:
                last_pan_feat = pan_cross_ms_feat
            if 'ms' in self.cross_module:
                last_ms_feat = ms_cross_pan_feat

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(last_pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(last_ms_feat)

        out = torch.cat([last_ms_feat, last_pan_feat], dim=1).reshape(b, -1)
        output = self.classifier(out)

        return output


def test_net():
    ms = torch.randn([20, 4, 16, 16]).to('cuda')
    pan = torch.randn([20, 1, 64, 64]).to('cuda')
    # CT_4 = CT_transform(4)
    # CT_1 = CT_transform(1)
    # ms_l, ms_s = CT_4.contourlet_decompose(ms)
    # pan_l1, pan_s1 = CT_1.contourlet_decompose(pan)
    # pan_l2, pan_s2 = CT_1.contourlet_decompose(pan_l1)
    # pan_l3, pan_s3 = CT_1.contourlet_decompose(pan_l2)
    net = CrossSwinTransformer(num_classes=8).to('cuda')
    y = net(ms, pan)
    print(y.shape)


if __name__ == '__main__':
    test_net()

