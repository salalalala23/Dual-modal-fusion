import torch.nn as nn
import torch
from model.swin_transformer import PatchEmbed, BasicLayer, PatchMerging

class Swin_T_dual(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed_m = PatchEmbed(
            patch_size=patch_size, in_c=4, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_p = PatchEmbed(
            patch_size=patch_size * 4, in_c=1, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers_m = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_m.append(layers)

        self.layers_p = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_p.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y):
        # x: [B, L, C]
        x, H1, W1 = self.patch_embed_m(x)
        x = self.pos_drop(x)

        # y, H2, W2 = self.patch_embed_p(y)
        # y = self.pos_drop(y)

        for layer in self.layers_m:
            x, H1, W1 = layer(x, H1, W1)

        # for layer in self.layers_p:
        #     y, H2, W2 = layer(y, H2, W2)

        x = self.norm(x)  # [B, L, C]
        # y = self.norm(y)
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def Net(args):
    model = Swin_T_dual(in_chans=4,
                        patch_size=args['trans']['patch_size'],
                        window_size=args['trans']['window_size'],
                        embed_dim=args['trans']['embed_dim'],
                        depths=args['trans']['depths'],
                        num_heads=args['trans']['num_heads'],
                        num_classes=args["Categories_Number"],
                        )
    return model


if __name__ == '__main__':
    m = torch.randn([20, 4, 16, 16])
    p = torch.randn([20, 1, 64, 64])
    args = {
        'Categories_Number': 12,
        "trans": {
            "patch_size": 4,
            "window_size": 2,
            "embed_dim": 96,
            "layers": 2,
            "depths": [6, 6],
            "num_heads": [3, 3],
        }
    }
    model = Net(args)
    y = model(m, p)
    print(y.shape)