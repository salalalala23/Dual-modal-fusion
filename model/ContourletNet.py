import torch
import torch.nn as nn
import pywt
import numpy as np

def get_filters():
    # 生成Contourlet变换所需的滤波器
    alpha = 0.75
    N = 6
    h = np.zeros((N, 2))
    g = np.zeros((N, 2))
    for i in range(N):
        h[i, 0] = np.sqrt(2) * np.cos(np.pi * (i + 1) / (2 * (N + 1)))
        h[i, 1] = np.sqrt(2) * np.sin(np.pi * (i + 1) / (2 * (N + 1)))
        g[i, 0] = np.sqrt(2) * np.cos(np.pi * i / (2 * (N + 1)))
        g[i, 1] = np.sqrt(2) * np.sin(np.pi * i / (2 * (N + 1)))
    h = alpha * h / np.linalg.norm(h)
    g = alpha * g / np.linalg.norm(g)
    return h, g

class ContourletTransform(nn.Module):
    # Contourlet变换模块
    def __init__(self):
        super(ContourletTransform, self).__init__()
        self.h, self.g = get_filters()

    def forward(self, x):
        c = []
        for i in range(x.shape[1]):
            # 对每个通道进行Contourlet变换
            c_channel = []
            for j in range(x.shape[0]):
                # 对每个尺度进行Contourlet变换
                coeffs2 = pywt.dwt2(x[j, i, :, :], 'db1')
                LL, (LH, HL, HH) = coeffs2
                c_scale = []
                for k in range(1, 5):
                    # 对每个方向进行Contourlet变换
                    coeffs2 = pywt.dwt2(LL, 'db1')
                    LL, (LH, HL, HH) = coeffs2
                    c_direction = []
                    for l in range(len(self.h)):
                        # 对每个子带进行Contourlet变换
                        coeffs2 = pywt.dwt2(LL, self.h[l])
                        LL, (LH, HL, HH) = coeffs2
                        coeffs2 = pywt.dwt2(LH, self.g[l])
                        LLH, (LHL, LHH, HHL, HHH) = coeffs2
                        coeffs2 = pywt.dwt2(HL, self.g[l])
                        HLH, (HLL, HLH, HHL, HHH) = coeffs2
                        coeffs2 = pywt.dwt2(HH, self.g[l])
                        HHH, (HHL, HLH, HLL, HHH) = coeffs2
                        c_subband = np.concatenate((LLH.ravel(), LHL.ravel(), LHH.ravel(), HLH.ravel(), HLL.ravel(), HHL.ravel(), HHH.ravel()))
                        c_direction.append(c_subband)
                    c_scale.append(np.concatenate(c_direction))
                c_channel.append(np.concatenate(c_scale))
            c.append(np.concatenate(c_channel))
        return torch.tensor(np.concatenate(c, axis=1)).float()

class ContourletNet(nn.Module):
    # 基于Contourlet变换的深度网络
    def __init__(self, num_classes=10):
        super(ContourletNet, self).__init__()
        self.contourlet = ContourletTransform()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.contourlet(x)
        x = x.view(-1, 18, 256, 256)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = x.view(-1, 128 * 32 * 32)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x


# 示例用法
net = ContourletNet(num_classes=10)
img = torch.randn([1, 3, 256, 256])
output = net(img)
print(output.shape)
