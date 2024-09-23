# from backbone import ResBlk
# from Mutual_Information2 import Mutual_Informations_18
# from Reduce_modal_Diff import Re_Mo_Diff_Loss
import torch
import torch.nn as  nn
import torch.nn.functional as F
# from Multimodal_Fusion import *
import utils.config as config
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
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=4,
                               stride=2, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
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
        out1 = F.relu(self.bn3(self.conv3(out)))
        out2 = F.relu(self.max_pooling(out))
        out = out1 + out2
        return out

# class ResNet18_All(nn.Module):
#     def __init__(self):
#         super(ResNet18_All, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Mutual_information = nn.ModuleList(Mutual_Informations_18())
#         self.Modal_Fusion_1 = Modal_Fusion()
#         self.Multi_Modal_Fusion = nn.ModuleList(Multi_Modal_Fusion_18())
#         self.Modal_Diff = Re_Mo_Diff_Loss(in_c=512)
#         self.outlayer = nn.Linear(512, 10)
#         # self.outlayer = nn.Linear(1024, 7)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         # # # Mutual_Information
#         # f_p_1, f_ms_1 = self.Mutual_information(f_p_1, f_ms_1)
#         # print('*********************************************************')
#         # f_p_2, f_ms_2 = self.Mutual_information(f_p_2, f_ms_2)
#         # print('*********************************************************')
#         # f_p_3, f_ms_3 = self.Mutual_information(f_p_3, f_ms_3)
#         # print('*********************************************************')
#         # f_p_4, f_ms_4 = self.Mutual_information(f_p_4, f_ms_4)
#         # print('*********************************************************')
#         # f_p_5, f_ms_5 = self.Mutual_information(f_p_5, f_ms_5)
#         # print('*********************************************************')
#         #
#         # # Mutual_Information
#         f_p_1, f_ms_1 = self.Mutual_information[0](f_p_1, f_ms_1)
#         f_p_2, f_ms_2 = self.Mutual_information[1](f_p_2, f_ms_2)
#         f_p_3, f_ms_3 = self.Mutual_information[2](f_p_3, f_ms_3)
#         f_p_4, f_ms_4 = self.Mutual_information[3](f_p_4, f_ms_4)
#         f_p_5, f_ms_5 = self.Mutual_information[4](f_p_5, f_ms_5)
#
#         # # Multi_Modal_Fusion
#         f_fu_1 = self.Modal_Fusion_1(f_p_1, f_ms_1)
#         f_fu_2 = self.Multi_Modal_Fusion[0](f_p_2, f_ms_2, f_fu_1)
#         f_fu_3 = self.Multi_Modal_Fusion[1](f_p_3, f_ms_3, f_fu_2)
#         f_fu_4 = self.Multi_Modal_Fusion[2](f_p_4, f_ms_4, f_fu_3)
#         f_fu_5 = self.Multi_Modal_Fusion[3](f_p_5, f_ms_5, f_fu_4)
#
#         out = []  # [loss_diff, rel] or [rel]
#         if phase == 'train':
#             # Loss
#             loss_diff = self.Modal_Diff(f_p_5, f_ms_5, f_i, f_hs)
#             out.append(loss_diff)
#
#         # f_ms_5  = F.adaptive_avg_pool2d(f_ms_5, [1, 1])
#         # f_p_5 = F.adaptive_avg_pool2d(f_p_5, [1, 1])
#         # s = torch.cat([f_ms_5, f_p_5], 1)  # 通道拼接
#         # s = s.view(s.size()[0], -1)
#         # rel = self.outlayer(s)
#         # out.append(rel)
#
#         rel = F.adaptive_avg_pool2d(f_fu_5, [1, 1])
#         rel = rel.view(rel.size()[0], -1)
#         rel = self.outlayer(rel)
#         out.append(rel)
#         return out

# class ResNet18_ori(nn.Module):
#     def __init__(self):
#         super(ResNet18_ori, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.outlayer = nn.Linear(1024, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         f_ms_5  = F.adaptive_avg_pool2d(f_ms_5, [1, 1])
#         f_p_5 = F.adaptive_avg_pool2d(f_p_5, [1, 1])
#         s = torch.cat([f_ms_5, f_p_5], 1)  # 通道拼接
#         s = s.view(s.size()[0], -1)
#         rel = self.outlayer(s)
#         return rel


# class ResNet18_MI(nn.Module):
#     def __init__(self):
#         super(ResNet18_MI, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Mutual_information = nn.ModuleList(Mutual_Informations_18())
#         self.outlayer = nn.Linear(1024, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_p_1, f_ms_1 = self.Mutual_information[0](f_p_1, f_ms_1)
#
#         f_p_2 = self.blk1_1(f_p_1)
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_p_2, f_ms_2 = self.Mutual_information[1](f_p_2, f_ms_2)
#
#         f_p_3 = self.blk2_1(f_p_2)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_p_3, f_ms_3 = self.Mutual_information[2](f_p_3, f_ms_3)
#
#         f_p_4 = self.blk3_1(f_p_3)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_p_4, f_ms_4 = self.Mutual_information[3](f_p_4, f_ms_4)
#
#         f_p_5 = self.blk4_1(f_p_4)
#         f_ms_5 = self.blk4_2(f_ms_4)
#         f_p_5, f_ms_5 = self.Mutual_information[4](f_p_5, f_ms_5)
#
#         f_ms_5  = F.adaptive_avg_pool2d(f_ms_5, [1, 1])
#         f_p_5 = F.adaptive_avg_pool2d(f_p_5, [1, 1])
#         s = torch.cat([f_ms_5, f_p_5], 1)  # 通道拼接
#         s = s.view(s.size()[0], -1)
#         rel = self.outlayer(s)
#
#         return rel


# class ResNet18_MF(nn.Module):
#     def __init__(self):
#         super(ResNet18_MF, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Modal_Fusion_1 = Modal_Fusion()
#         self.Multi_Modal_Fusion = nn.ModuleList(Multi_Modal_Fusion_18())
#         self.outlayer = nn.Linear(512, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         # # Multi_Modal_Fusion
#         f_fu_1 = self.Modal_Fusion_1(f_p_1, f_ms_1)
#         f_fu_2 = self.Multi_Modal_Fusion[0](f_p_2, f_ms_2, f_fu_1)
#         f_fu_3 = self.Multi_Modal_Fusion[1](f_p_3, f_ms_3, f_fu_2)
#         f_fu_4 = self.Multi_Modal_Fusion[2](f_p_4, f_ms_4, f_fu_3)
#         f_fu_5 = self.Multi_Modal_Fusion[3](f_p_5, f_ms_5, f_fu_4)
#
#         rel = F.adaptive_avg_pool2d(f_fu_5, [1, 1])
#         rel = rel.view(rel.size()[0], -1)
#         rel = self.outlayer(rel)
#         return rel


# class ResNet18_Diff(nn.Module):
#     def __init__(self):
#         super(ResNet18_Diff, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Modal_Diff = Re_Mo_Diff_Loss(in_c=512)
#         self.outlayer = nn.Linear(1024, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         out = []  # [loss_diff, rel] or [rel]
#         if phase == 'train':
#             # Loss
#             loss_diff = self.Modal_Diff(f_p_5, f_ms_5, f_i, f_hs)
#             out.append(loss_diff)
#
#         f_ms_5  = F.adaptive_avg_pool2d(f_ms_5, [1, 1])
#         f_p_5 = F.adaptive_avg_pool2d(f_p_5, [1, 1])
#         s = torch.cat([f_ms_5, f_p_5], 1)  # 通道拼接
#         s = s.view(s.size()[0], -1)
#         rel = self.outlayer(s)
#         out.append(rel)
#         return out


# class ResNet18_MF_Diff(nn.Module):
#     def __init__(self):
#         super(ResNet18_MF_Diff, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Modal_Fusion_1 = Modal_Fusion()
#         self.Multi_Modal_Fusion = nn.ModuleList(Multi_Modal_Fusion_18())
#         self.Modal_Diff = Re_Mo_Diff_Loss(in_c=512)
#         self.outlayer = nn.Linear(512, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         # # Multi_Modal_Fusion
#         f_fu_1 = self.Modal_Fusion_1(f_p_1, f_ms_1)
#         f_fu_2 = self.Multi_Modal_Fusion[0](f_p_2, f_ms_2, f_fu_1)
#         f_fu_3 = self.Multi_Modal_Fusion[1](f_p_3, f_ms_3, f_fu_2)
#         f_fu_4 = self.Multi_Modal_Fusion[2](f_p_4, f_ms_4, f_fu_3)
#         f_fu_5 = self.Multi_Modal_Fusion[3](f_p_5, f_ms_5, f_fu_4)
#
#         out = []  # [loss_diff, rel] or [rel]
#         if phase == 'train':
#             # Loss
#             loss_diff = self.Modal_Diff(f_p_5, f_ms_5, f_i, f_hs)
#             out.append(loss_diff)
#
#         rel = F.adaptive_avg_pool2d(f_fu_5, [1, 1])
#         rel = rel.view(rel.size()[0], -1)
#         rel = self.outlayer(rel)
#         out.append(rel)
#         return out


# class ResNet18_MI_Diff(nn.Module):
#     def __init__(self):
#         super(ResNet18_MI_Diff, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Mutual_information = nn.ModuleList(Mutual_Informations_18())
#         self.Modal_Fusion_1 = Modal_Fusion()
#         self.Modal_Diff = Re_Mo_Diff_Loss(in_c=512)
#         self.outlayer = nn.Linear(1024, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_p_1, f_ms_1 = self.Mutual_information[0](f_p_1, f_ms_1)
#
#         f_p_2 = self.blk1_1(f_p_1)
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_p_2, f_ms_2 = self.Mutual_information[1](f_p_2, f_ms_2)
#
#         f_p_3 = self.blk2_1(f_p_2)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_p_3, f_ms_3 = self.Mutual_information[2](f_p_3, f_ms_3)
#
#         f_p_4 = self.blk3_1(f_p_3)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_p_4, f_ms_4 = self.Mutual_information[3](f_p_4, f_ms_4)
#
#         f_p_5 = self.blk4_1(f_p_4)
#         f_ms_5 = self.blk4_2(f_ms_4)
#         f_p_5, f_ms_5 = self.Mutual_information[4](f_p_5, f_ms_5)
#
#         out = []  # [loss_diff, rel] or [rel]
#         if phase == 'train':
#             # Loss
#             loss_diff = self.Modal_Diff(f_p_5, f_ms_5, f_i, f_hs)
#             out.append(loss_diff)
#
#         f_ms_5  = F.adaptive_avg_pool2d(f_ms_5, [1, 1])
#         f_p_5 = F.adaptive_avg_pool2d(f_p_5, [1, 1])
#         s = torch.cat([f_ms_5, f_p_5], 1)  # 通道拼接
#         s = s.view(s.size()[0], -1)
#         rel = self.outlayer(s)
#         out.append(rel)
#
#         return out


# class ResNet18_MI_MF(nn.Module):
#     def __init__(self):
#         super(ResNet18_MI_MF, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         self.blk1_1 = ResBlk(64, 64, stride=1)
#         self.blk2_1 = ResBlk(64, 128, stride=1)
#         self.blk3_1 = ResBlk(128, 256, stride=1)
#         self.blk4_1 = ResBlk(256, 512, stride=1)
#
#         self.blk1_2 = ResBlk(64, 64, stride=1)
#         self.blk2_2 = ResBlk(64, 128, stride=1)
#         self.blk3_2 = ResBlk(128, 256, stride=1)
#         self.blk4_2 = ResBlk(256, 512, stride=1)
#
#         self.Mutual_information = nn.ModuleList(Mutual_Informations_18())
#         self.Modal_Fusion_1 = Modal_Fusion()
#         self.Multi_Modal_Fusion = nn.ModuleList(Multi_Modal_Fusion_18())
#         self.outlayer = nn.Linear(512, 10)
#
#     def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
#         """
#         input:
#             f_p:[B,1,64,64]
#             f_ms:[B,4,64,64]
#             f_i:[B, 1, 64, 64]
#             f_hs:[B, 2, 16, 16]
#         output:
#             rel:[B,7]
#         """
#         # PAN支路
#         f_p_1 = F.relu(self.conv1(f_p))
#         f_p_2 = self.blk1_1(f_p_1)
#         f_p_3 = self.blk2_1(f_p_2)
#         f_p_4 = self.blk3_1(f_p_3)
#         f_p_5 = self.blk4_1(f_p_4)
#
#         # MS支路
#         f_ms_1 = F.relu(self.conv2(f_ms))
#         f_ms_2 = self.blk1_2(f_ms_1)
#         f_ms_3 = self.blk2_2(f_ms_2)
#         f_ms_4 = self.blk3_2(f_ms_3)
#         f_ms_5 = self.blk4_2(f_ms_4)
#
#         # # Mutual_Information
#         f_p_1, f_ms_1 = self.Mutual_information[0](f_p_1, f_ms_1)
#         f_p_2, f_ms_2 = self.Mutual_information[1](f_p_2, f_ms_2)
#         f_p_3, f_ms_3 = self.Mutual_information[2](f_p_3, f_ms_3)
#         f_p_4, f_ms_4 = self.Mutual_information[3](f_p_4, f_ms_4)
#         f_p_5, f_ms_5 = self.Mutual_information[4](f_p_5, f_ms_5)
#
#         # # Multi_Modal_Fusion
#         f_fu_1 = self.Modal_Fusion_1(f_p_1, f_ms_1)
#         f_fu_2 = self.Multi_Modal_Fusion[0](f_p_2, f_ms_2, f_fu_1)
#         f_fu_3 = self.Multi_Modal_Fusion[1](f_p_3, f_ms_3, f_fu_2)
#         f_fu_4 = self.Multi_Modal_Fusion[2](f_p_4, f_ms_4, f_fu_3)
#         f_fu_5 = self.Multi_Modal_Fusion[3](f_p_5, f_ms_5, f_fu_4)
#
#         rel = F.adaptive_avg_pool2d(f_fu_5, [1, 1])
#         rel = rel.view(rel.size()[0], -1)
#         rel = self.outlayer(rel)
#
#         return rel


class ResNet18_CRHFF(nn.Module):
    def __init__(self):
        super(ResNet18_CRHFF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=1)
        self.blk3_2 = ResBlk(128, 256, stride=1)
        self.blk4_2 = ResBlk(256, 512, stride=1)

        self.outlayer = nn.Linear(640, Categories)

    def forward(self, f_ms, f_p, f_i=0, f_hs=0, phase='train'):
        """
        input:
            f_p:[B,1,64,64]
            f_ms:[B,4,16,16]
            f_i:[B, 1, 64, 64]
            f_hs:[B, 2, 16, 16]
        output:
            rel:[B,7]
        """
        # PAN支路
        f_p_1 = F.relu(self.conv1(f_p))    # [B. 64, 64, 64]

        # MS支路
        f_ms_1 = F.relu(self.conv2(f_ms))   # [B, 64, 16,16]
        f_ms_2 = self.blk1_2(f_ms_1)
        f_ms_2_up = nn.functional.interpolate(f_ms_2, size=(64, 64))
        f_ms_3 = self.blk2_2(f_ms_2)
        f_ms_4 = self.blk3_2(f_ms_3)
        f_ms_5 = self.blk4_2(f_ms_4)
        f_ms_5_up = nn.functional.interpolate(f_ms_5, size=(64, 64))

        f_fu = torch.cat([f_ms_2_up, f_ms_5_up, f_p_1], 1)    # [B, 640, 64, 64]
        s = F.adaptive_avg_pool2d(f_fu, [1, 1])
        s = s.view(s.size()[0], -1)
        rel = self.outlayer(s)
        return rel