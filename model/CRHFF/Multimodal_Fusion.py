import cv2
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
# from contourlet import filters, operations, sampling, transform
# from Mutual_Information import Similarity


class Modal_Fusion(nn.Module):
    def __init__(self, input_c_1=64, input_c_2=64):
        super(Modal_Fusion, self).__init__()
        self.softmax = nn.Sigmoid()
        self.maxpooling = nn.AdaptiveMaxPool2d(output_size=(16, 16))
        self.layer = nn.Sequential(
            nn.BatchNorm2d(input_c_1),
            nn.Conv2d(in_channels=input_c_1, out_channels=input_c_1, kernel_size=1),
            # nn.BatchNorm2d(input_c_1),
            nn.ReLU(inplace=True)
        )
        self.extra = nn.Sequential()
        if input_c_1 != input_c_2:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels=input_c_2, out_channels=input_c_1, kernel_size=1),
                nn.BatchNorm2d(input_c_1),
                nn.ReLU(inplace=True)
            )

    def forward(self, f_p, f_ms):
        """
        input:
            f_p:[B,C,64,64] or [B,C,16,16]
            f_ms:[B,C,16,16]
        output:
            rel:[B,C,16,16]
        """
        s_p = f_p.size()[2]
        s_ms = f_ms.size()[2]
        f_ms = self.extra(f_ms)
        # s_m = Similarity(f_p, f_ms)   # [B, C, C]
        # s_m_ms = 1. - self.softmax(torch.mean(s_m, dim=2))   # [B, C]   # 或不减
        # s_m_p = 1. - self.softmax(torch.mean(s_m, dim=1))  # [B, C]   # 或不减

        # s_m_ms = self.softmax(torch.mean(s_m, dim=2))  # [B, C]   # 或不减
        # s_m_p =  self.softmax(torch.mean(s_m, dim=1))  # [B, C]   # 或不减
        # s_m_ms = torch.unsqueeze(torch.unsqueeze(s_m_ms, 2), 3)
        # s_m_p = torch.unsqueeze(torch.unsqueeze(s_m_p, 2), 3)  # [B, C, 1, 1]

        # f_p_t = torch.add(f_p, torch.mul(f_p, s_m_p))    # [B,C,64,64]
        # f_ms_t = torch.add(f_ms, torch.mul(f_ms, s_m_ms))  # [B,C,16,16]

        # if s_ms != s_p:
        #     f_p_t = self.maxpooling(f_p_t)   # [B, C, 16, 16]   # 如果都是16，这一步不用
        # f_fu = torch.add(f_p_t, f_ms_t)   # [B, C, 16, 16]

        if s_ms != s_p:
            f_p = self.maxpooling(f_p)   # [B, C, 16, 16]   # 如果都是16，这一步不用
        f_fu = torch.add(f_p, f_ms)   # [B, C, 16, 16]
        f_fu = self.layer(f_fu)     # [B, C, 16, 16]
        return f_fu


class Multi_Modal_Fusion(nn.Module):
    def __init__(self, in_p=256, in_ms=256, in_3=64):  # (64, 64, 64), (128, 128, 64), (256, 256, 128), (512, 512, 256)
        super(Multi_Modal_Fusion, self).__init__()
        self.MF1 = Modal_Fusion(input_c_1=in_p, input_c_2=in_ms)
        self.MF2 = Modal_Fusion(input_c_1=in_p, input_c_2=in_3)
        self.MF3 = Modal_Fusion(input_c_1=in_ms, input_c_2=in_3)
        self.change_c = nn.Sequential(
            # nn.BatchNorm2d(in_p),
            nn.Conv2d(in_channels=in_p * 3, out_channels=in_p, kernel_size=1),
            nn.BatchNorm2d(in_p),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_p, f_ms, f_u):
        """
        input:
            f_p:[B,C,64,64]
            f_ms:[B,C,16,16]
            f_u:[B, C//2, 16, 16]
        output:
            rel:[B,C,16,16]
        """
        f_fu1 = self.MF1(f_p, f_ms)
        f_fu2 = self.MF2(f_p, f_u)
        f_fu1 = torch.cat([f_fu1, f_fu2], dim=1)
        f_fu3 = self.MF3(f_ms, f_u)
        f_fu = torch.cat([f_fu3, f_fu1], dim=1)
        rel = self.change_c(f_fu)
        return rel


def Multi_Modal_Fusion_18():
    cfg_18 = [(64, 64, 64), (128, 128, 64), (256, 256, 128), (512, 512, 256)]
    Multi_Modal_Fusion_18 = []
    for i in range(len(cfg_18)):
        m = Multi_Modal_Fusion(in_p=cfg_18[i][0], in_ms=cfg_18[i][1], in_3=cfg_18[i][2])
        Multi_Modal_Fusion_18.append(m)
    return Multi_Modal_Fusion_18


def Multi_Modal_Fusion_50():
    cfg_50 = [(128, 128, 64), (256, 256, 128), (512, 512, 256), (1024, 1024, 512)]
    Multi_Modal_Fusion_50 = []
    for i in range(len(cfg_50)):
        m = Multi_Modal_Fusion(in_p=cfg_50[i][0], in_ms=cfg_50[i][1], in_3=cfg_50[i][2])
        Multi_Modal_Fusion_50.append(m)
    return Multi_Modal_Fusion_50


def test():
    a = torch.randn(2, 64, 64, 64)
    b = torch.randn(2, 56, 16, 16)
    M = Modal_Fusion(64, 56)
    print(M(a, b).size())

def test2():
    a = torch.randn(2, 64, 64, 64)
    c = nn.Conv2d(64, 64, kernel_size=1)
    print(c(a).size())

# test2()


# a = torch.randn(2, 12, 64, 64)
# b = torch.randn(2, 12, 16, 16)
# c = torch.randn(2, 5, 16, 16)
# M = Multi_Modal_Fusion(in_p=12, in_ms=12, in_3=5)
# m = M(a, b, c)
# print(m.size())
