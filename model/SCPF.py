import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch


class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_out!=ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.extra(x)+out
        out = F.relu(out)

        return out
# class SEBlock(nn.Module):

#     def __init__(self):
#         super(SEBlock, self).__init__()
#         self.relu = nn.ReLU(inplace=False)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # N * 32 * 1 * 1
#         self.fc1 = nn.Linear(in_features=256, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=256)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # sequeeze
#         out = self.global_pool(x)   
#         out = out.view(out.size(0), -1)
#         # Excitation
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         out = out.view(out.size(0), out.size(1), 1, 1)
#         # Scale
#         out = out * x
#         out += x
#         out = self.relu(out)

#         return out


class ABlock(nn.Module):
    
    def __init__(self, planes = 32):
        super(ABlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes // 2, 1, 1, 0)      # 32 * 32 * 32
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, 1, 1, 0) # 32 * 32 * 32
        self.conv3 = nn.Conv2d(planes // 2, planes // 2, 3, 1, 1) # 32 * 32 * 32
        self.conv4 = nn.Conv2d(planes // 2, planes // 2, 5, 1, 2) # 32 * 32 * 32
        self.sigmoid = nn.Sigmoid()
        self.conv5 = nn.Conv2d(1, planes // 4, 3, 1, 1)
        self.conv6 = nn.Conv2d(planes // 4, 1, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # N * 256 * 1 * 1
        self.conv7 = nn.Conv2d(planes*2, planes, 3, 1, 1)


    def forward(self, x):
        F0 = self.conv1(x)
        F1 = self.conv2(F0)
        F2 = self.conv3(F0)
        F3 = self.conv4(F0)

        

        a0 = torch.mean(F0, dim=1,keepdim = True)
        a1 = torch.mean(F1, dim=1,keepdim = True)
        a2 = torch.mean(F2, dim=1,keepdim = True)
        a3 = torch.mean(F3, dim=1,keepdim = True)

        a0 = self.sigmoid(self.conv6(F.relu(self.conv5(a0))))
        a1 = self.sigmoid(self.conv6(F.relu(self.conv5(a1))))
        a2 = self.sigmoid(self.conv6(F.relu(self.conv5(a2))))
        a3 = self.sigmoid(self.conv6(F.relu(self.conv5(a3))))

        # a0 = torch.div(torch.sum(F0, 1, keepdim=True), F0.shape[1])
        # a1 = torch.div(torch.sum(F1, 1, keepdim=True), F1.shape[1])
        # a2 = torch.div(torch.sum(F2, 1, keepdim=True), F2.shape[1])
        # a3 = torch.div(torch.sum(F3, 1, keepdim=True), F3.shape[1])
        a0_int = self.global_pool(a0) # N * 32 * 1 * 1
        a1_int = self.global_pool(a1) # N * 32 * 1 * 1
        a2_int = self.global_pool(a2) # N * 32 * 1 * 1
        a3_int = self.global_pool(a3) # N * 32 * 1 * 1
        # a0_int = torch.div(torch.sum(a0,(2, 3)), a0.shape[2] * a0.shape[3])
        # a1_int = torch.div(torch.sum(a1,(2, 3)), a1.shape[2] * a1.shape[3])
        # a2_int = torch.div(torch.sum(a2,(2, 3)), a2.shape[2] * a2.shape[3])
        # a3_int = torch.div(torch.sum(a3,(2, 3)), a3.shape[2] * a3.shape[3])
        a_int = torch.cat([a0_int, a1_int, a2_int, a3_int], 1)  # 构建N * 4 的 tensor
        a_sort = torch.sort(a_int)  # 挑选出最大的两个系数
        F_0 = torch.mul(F0, a0)
        F_1 = torch.mul(F1, a1)
        F_2 = torch.mul(F2, a2)
        F_3 = torch.mul(F3, a3)
        # S1 = torch.zeros(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        # S2 = torch.zeros(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        # # 得到B模块的输入
        # for i in range(a_sort[1].shape[0]):
        #     if i == 0:
        #         S1 = (locals()['F_' + str(a_sort[1][i][3].item())][i]).view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        #         S2 = (locals()['F_' + str(a_sort[1][i][2].item())][i]).view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        #     else:
        #         S1 = torch.cat([ locals()['F_' + str(a_sort[1][i][3].item())][i].view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2]), S1 ], 0)
        #         S2 = torch.cat([ locals()['F_' + str(a_sort[1][i][2].item())][i].view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2]), S2 ], 0)

        # # 拼接S1， S2
        S = torch.cat([F_0, F_1,F_2,F_3], 1) # 通道拼接
        S = self.conv7(S)
        S = S + x

        return F.relu(S)


class BBlock(nn.Module):

    def __init__(self, out_channel=32):
        super(BBlock, self).__init__()
        self.M = 2
        self.out_channels = out_channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # N * 256 * 1 * 1
        self.fc1 = nn.Sequential(nn.Conv2d(self.out_channels, 8, 1, bias=False),
                               nn.BatchNorm2d(8),
                               nn.ReLU(inplace=False))    # 256通道降维到64通道
        self.fc2 = nn.Conv2d(8, self.out_channels, 1, 1, bias=False) # 8通道 升维到32通道
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        output = []
        # the part of split
        x1, x2 = torch.chunk(x, 2, 1) # 将x沿第2个维度分成两份
        output.append(x1) # N * 16 * 32 * 32
        output.append(x2) # N * 16 * 32 * 32
        # the part of fusion
        Vf = self.global_pool(x) # N * 32 * 1 * 1
        z=self.fc1(Vf)           # N * 8 * 1 * 1
        a_b = self.fc2(z)        # N * 32 * 1 * 1
        a_b = a_b.reshape(batch_size, self.M, self.out_channels // 2, -1) # N * 2 * 16 * 1 
        a_b = self.softmax(a_b) # N * 2 * 16 * 1 
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1)) # N * 1 * 16 * 1
        a_b = list(map(lambda x:x.reshape(batch_size, self.out_channels // 2, 1, 1), a_b)) # N * 16 * 1 * 1
        V = list(map(lambda x, y:x*y, output, a_b)) # 2个 N * 16 * 32 * 32
        O = torch.cat([V[0], V[1]], 1) # 通道拼接
        return F.relu(O)


class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18,self).__init__()
        self.conv1_1= nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        # [b,64,h,w]=>[b,128,h,w]
        self.blk1_1 = ResBlk(16,24)
        self.blk2_1 = ResBlk(16,24)
        self.blk3_1 = ResBlk(16,24)
        # [b,128,h,w]=>[b,256,h,w]
        self.blk1_2 = ResBlk(24,32)
        self.blk2_2 = ResBlk(24,32)
        self.blk3_2 = ResBlk(24,32)
        # [b,256,h,w]=>[b,512,h,w]
        self.blk1_3 = ResBlk(64,64)
        self.blk2_3 = ResBlk(64,64)
        # [b,512,h,w]=>[b,512,h,w]
        self.blk1_4 = ResBlk(64,64)
        self.blk2_4 = ResBlk(64,64)

        self.blk1_5 = ResBlk(64,64)
        self.blk2_5 = ResBlk(64,64)

        self.blk6 = ResBlk(128,128)

        self.Ablock_64  = ABlock(32)
        self.Bblock_64  = BBlock(32)

        self.RP1_1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 8 * 32 * 32

        self.RP2_1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 8 * 32 * 32

        self.RP1_2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16

        self.RP2_2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16
        
        self.R4 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16
        self.R5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16

        self.R6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16

        self.RP6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) # N * 16 * 16 * 16

        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=args['Categories_Number'])
    
    def forward(self, ms, pan, mshpan):
        
        # PAN_fusion
        
        out_top = self.RP1_1(pan)
        out_top = self.RP1_2(out_top)
        out_top = F.relu(self.conv1_1(out_top))
        out_top = self.blk1_1(out_top)
        # out_top = self.Ablock_64(out_top)
        # out_top = self.Bblock_64(out_top)
        out_top = self.blk1_2(out_top)
        # out_top1 = out_top
        # out_top_A = self.Ablock_64(out_top)
        # out_top_B = self.Bblock_64(out_top)


        # PAN + MS_H
        out_center = self.RP2_1(mshpan)
        out_center = self.RP2_2(out_center)
        out_center = F.relu(self.conv1_2(out_center))
        out_center = self.blk2_1(out_center)
        # out_center = self.Ablock_64(out_center)
        # out_center = self.Bblock_64(out_center)
        out_center = self.blk2_2(out_center)
        out_center_A = self.Ablock_64(out_center)
        # out_center_A = out_center
        # out_center_top = out_center # 与PAN_fusion结合
        out_center_B = self.Bblock_64(out_center)
        
        #　MS_fusion
        out_bottom = self.R4(ms)
        out_bottom = F.relu(self.conv1_3(out_bottom))
        out_bottom = self.blk3_1(out_bottom)
        # out_bottom = self.Ablock_64(out_bottom)
        # out_bottom = self.Bblock_64(out_bottom)
        out_bottom = self.blk3_2(out_bottom)
        # out_bottom1 = out_bottom
        
        # out_bottom_A  = self.Ablock_64(out_bottom)

        # out_bottom_B = self.Bblock_64(out_bottom)

        # middle fusion
        # input_blk3_top = torch.cat([out_top,out_top,out_center_top,out_center_top], 1)   # 深度通道拼接
        input_blk3_top = torch.cat([out_top,out_center_A], 1)   # 深度通道拼接
        out_blk3_top = self.blk1_3(input_blk3_top)
        out_blk4_top = self.blk1_4(out_blk3_top)
        out_blk5_top = self.blk1_5(out_blk4_top)
        # input_B_top = self.Ablock_128(out_blk3_top)
        # out_B_top = self.Bblock_128(input_B_top)

        # input_blk3_bottom = torch.cat([out_center,out_center,out_bottom,out_bottom], 1)   # 深度通道拼接
        input_blk3_bottom = torch.cat([out_center_B,out_bottom], 1)   # 深度通道拼接
        out_blk3_bottom = self.blk2_3(input_blk3_bottom)
        out_blk4_bottom = self.blk2_4(out_blk3_bottom)
        out_blk5_bottom = self.blk2_5(out_blk4_bottom)
        # input_B_bottom= self.Ablock_128(out_blk3_bottom)
        # out_B_bottom = self.Bblock_128(input_B_bottom)

        # deep fusion
        # input_blk4 = torch.cat([out_blk3_top,out_blk3_top,out_blk3_bottom,out_blk3_bottom], 1)   # 深度通道拼接
        input_blk6 = torch.cat([out_blk5_top,out_blk5_bottom], 1)   # 深度通道拼接
        out_blk6 = self.blk6(input_blk6)
        out_blk6 = self.R5(out_blk6)
        out_blk6 = self.R6(out_blk6)
        # out_blk4 = self.RP6(out_blk4)

        # 全连接分类
        input_FC = out_blk6.view(out_blk6.shape[0], -1)
        input_FC = F.relu(self.fc1(input_FC))
        input_FC = F.relu(self.fc2(input_FC))
        out_FC = self.fc3(input_FC)

        return out_FC


def Net(args):
    return ResNet18(args)


def test_net():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ms = torch.randn([20, 4, 16, 16]).to(device)
    # ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 8,
        'device': 'cuda:0'
    }
    net = Net(cfg).to(device)
    y = net(ms, pan)
    print(y.shape)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='N_SR')
    # parser.add_argument('--Categories', type=int, default=12)
    # args = parser.parse_args()
    # pan = torch.randn(2, 1, 64, 64)
    # ms = torch.randn(2, 4, 16, 16)
    # mshpan = torch.randn(2, 1, 64, 64)
    # grf_net = ResNet18(args)
    # out_result = grf_net(ms,pan,mshpan)
    # print(out_result)
    # print(out_result.shape)
    test_net()
