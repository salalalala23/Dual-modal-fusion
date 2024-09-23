from Multimodal_Fusion import *
# import thop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import config
Categories = config.Categories


class ResBlk(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_MF(nn.Module):
    def __init__(self):
        super(ResNet18_MF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1_1 = ResBlk(64, 64, stride=1)
        self.blk2_1 = ResBlk(64, 128, stride=1)
        self.blk3_1 = ResBlk(128, 256, stride=1)
        self.blk4_1 = ResBlk(256, 512, stride=1)

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=1)
        self.blk3_2 = ResBlk(128, 256, stride=1)
        self.blk4_2 = ResBlk(256, 512, stride=1)

        self.Modal_Fusion_1 = Modal_Fusion()
        self.Multi_Modal_Fusion = nn.ModuleList(Multi_Modal_Fusion_18())
        self.outlayer = nn.Linear(512, Categories)

    def forward(self, f_ms, f_p, f_i=0, f_hs=0, phase='train'):
        """
        input:
            f_p:[B,1,64,64]
            f_ms:[B,4,64,64]
            f_i:[B, 1, 64, 64]
            f_hs:[B, 2, 16, 16]
        output:
            rel:[B,7]
        """
        # PAN支路
        f_p_1 = F.relu(self.conv1(f_p))  # [20, 64, 64, 64]
        f_p_2 = self.blk1_1(f_p_1)  # [20, 64, 64, 64]
        f_p_3 = self.blk2_1(f_p_2)  # [20, 128, 64, 64]
        f_p_4 = self.blk3_1(f_p_3)  # [20, 256, 64, 64]F_
        f_p_5 = self.blk4_1(f_p_4)  # [20, 512, 64, 64]

        # MS支路
        f_ms_1 = F.relu(self.conv2(f_ms))  # [20, 64, 16, 16]
        f_ms_2 = self.blk1_2(f_ms_1)  # [20, 64, 16, 16]
        f_ms_3 = self.blk2_2(f_ms_2)  # [20, 64, 16, 16]
        f_ms_4 = self.blk3_2(f_ms_3)  # [20, 256, 16, 16]
        f_ms_5 = self.blk4_2(f_ms_4)  # [20, 512, 16, 16]

        # # 外部特征支路
        # g_p_1 = F.relu(self.conv3(f_p))  # [20, 3, 64, 64]
        # g_ms_1 = F.relu(self.conv4(f_ms))  # [20, 3, 16, 16]
        # g_p_l1, g_p_s1 = transform.contourlet_decompose(g_p_1)  # [20, 3, 32, 32], [20, 12, 32, 32]
        # g_ms_l, g_ms_s = transform.contourlet_decompose(g_ms_1)  # [20, 3, 8, 8], [20, 12, 8, 8]
        # g_p_l2, g_p_s2 = transform.contourlet_decompose(g_p_l1)  # [20, 3, 16, 16], [20, 12, 16, 16]
        # g_p_l3, g_p_s3 = transform.contourlet_decompose(g_p_l2)  # [20, 3, 8, 8], [20, 12, 8, 8]
        # g_fu_l1 = transform.contourlet_recompose(g_p_l3+g_ms_l,g_ms_s+g_p_s3)  # [20, 3, 16, 16]
        # g_fu_l2 = transform.contourlet_recompose(g_fu_l1, g_p_s2)  # [20, 3, 32, 32]
        # g_fu_l3 = transform.contourlet_recompose(g_fu_l2, g_p_s1)  # [20, 3, 64, 64]
        # g_fu_1 = F.relu(self.conv5(g_fu_l3))  # [20, 64, 64, 64]
        # g_fu_2 = self.blk1_1(g_fu_1)  # [20, 64, 64, 64]
        # g_fu_3 = self.blk2_1(g_fu_2)  # [20, 128, 64, 64]
        # g_fu_4 = self.blk3_1(g_fu_3)  # [20, 256, 64, 64]
        # g_fu_5 = self.blk4_1(g_fu_4)  # [20, 512, 64, 64]
        # print(g_fu_5.shape)
        # rel1 = F.adaptive_avg_pool2d(g_fu_5, [1, 1])  # [20, 512, 1, 1]
        # rel1 = rel1.view(rel1.size()[0], -1)  # [20, 512]
        # rel1 = self.outlayer(rel1)  # [20, 10]
        # # Multi_Modal_Fusion
        f_fu_1 = self.Modal_Fusion_1(f_p_1, f_ms_1)  # [20, 64, 16, 16]
        f_fu_2 = self.Multi_Modal_Fusion[0](f_p_2, f_ms_2, f_fu_1)  # [20, 64, 16, 16]
        f_fu_3 = self.Multi_Modal_Fusion[1](f_p_3, f_ms_3, f_fu_2)  # [20, 128, 16, 16]
        f_fu_4 = self.Multi_Modal_Fusion[2](f_p_4, f_ms_4, f_fu_3)  # [20, 256, 16, 16]
        f_fu_5 = self.Multi_Modal_Fusion[3](f_p_5, f_ms_5, f_fu_4)  # [20, 512, 16, 16]

        rel2 = F.adaptive_avg_pool2d(f_fu_5, [1, 1])  # [20, 512, 1, 1]
        rel2 = rel2.view(rel2.size()[0], -1)  # [20, 512]
        rel2 = self.outlayer(rel2)  # [20, 10]
        return rel2


def test():
    model = ResNet18_MF()
    model = model.to(device)
    a = torch.randn(20, 4, 16, 16).to(device)
    b = torch.randn(20, 1, 64, 64).to(device)
    # flops, params = thop.profile(model, inputs=(a,b))
    # # 多输入则为
    # # flops, params = thop.profile(model, inputs=(x, y, z))
    # flops, params = thop.clever_format([flops, params], '%.3f')
    # print('flops:', flops)
    # print('params:', params)
    y = model(a, b)
    print(y.size())


if __name__ == '__main__':
    test()