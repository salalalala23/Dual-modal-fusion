from torch.utils.data import Dataset
from function.function import to_tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


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
        img = img.to(device)
        # channel = img.shape[1]
        # 9-7 filters
        # h, g = filters.lp_filters(channel)
        # Laplacian Pyramid decompose
        low_band, high = self.lp_dec(img)
        # save_image(low_band, 'test_image/low_band'+index+'.png')
        # save_image(high, 'test_image/high_band'+index+'.png')
        # DFB filters
        # h0, h1 = self.dfb_filters(channel, mode='d', name=name)
        # DFB decompose
        sub_bands = self.dfb_dec(high, name=name)
        return low_band, sub_bands


class dataset_one(Dataset):
    def __init__(self, ms, label, x, y, size):
        self.MS = ms
        self.Label = label
        self.x = x
        self.y = y
        self.ms_size = size

    def __getitem__(self, index):
        ms_size = self.ms_size
        ms_x = int(self.x[index])
        ms_y = int(self.y[index])
        pan_x = int(4 * ms_x)
        pan_y = int(4 * ms_y)
        image_ms = self.MS[ms_x:ms_x+ms_size, ms_y:ms_y+ms_size, :]
        label = torch.Tensor(self.Label[index])
        label = label.squeeze()
        image_ms = image_ms.transpose((2, 0, 1))
        image_ms = torch.from_numpy(image_ms).type(torch.FloatTensor)
        return image_ms, label, ms_x, ms_y

    def __len__(self):
        return len(self.x)


class dataset_dual(Dataset):
    def __init__(self, ms, pan, xyl, cfg):  # 2, 0, 1
        self.MS = ms
        self.PAN = pan
        self.Label = xyl[2]
        self.x = xyl[0]
        self.y = xyl[1]
        self.ms_size = cfg['patch_size']
        self.pan_size = cfg['patch_size'] * 4

    def __getitem__(self, index):
        ms_size = self.ms_size
        pan_size = self.pan_size
        ms_x = int(self.x[index])
        ms_y = int(self.y[index])
        pan_x = int(4 * ms_x)
        pan_y = int(4 * ms_y)
        image_ms = self.MS[ms_x:ms_x+ms_size, ms_y:ms_y+ms_size, :]
        image_pan = self.PAN[pan_x:pan_x + pan_size, pan_y:pan_y + pan_size]
        label = torch.Tensor(self.Label[index])
        label = label.squeeze()
        image_ms = image_ms.transpose((2, 0, 1))
        image_pan = np.expand_dims(image_pan, axis=0)
        # image_ms = image_ms.astype(float)
        # image_pan = image_pan.astype(float)
        image_ms = torch.from_numpy(image_ms).type(torch.FloatTensor)
        image_pan = torch.from_numpy(image_pan).type(torch.FloatTensor)
        return image_ms, image_pan, label, ms_x, ms_y

    def __len__(self):
        return len(self.x)


class dataset_qua_dqtl(Dataset):
    def __init__(self, ms, pan, ms_gan, pan_gan, xyl, cfg):  # 2, 0, 1
        self.MS = ms
        self.PAN = pan
        self.ms_gan = ms_gan
        self.pan_gan = pan_gan
        self.Label = xyl[2]
        self.x = xyl[0]
        self.y = xyl[1]
        self.size = cfg['patch_size']

    def __getitem__(self, index):
        size = self.size
        x, y = int(self.x[index]), int(self.y[index])
        i_ms = self.MS[x:x+size, y:y+size, :]
        i_pan = self.PAN[x:x+size, y:y+size, :]
        i_ms_gan = self.ms_gan[x:x + size, y:y + size, :]
        i_pan_gan = self.pan_gan[x:x + size, y:y + size, :]
        label = torch.Tensor(self.Label[index])
        label = label.squeeze()
        image_ms = i_ms.transpose((2, 0, 1))
        image_pan = i_pan.transpose((2, 0, 1))
        image_ms_gan = i_ms_gan.transpose((2, 0, 1))
        image_pan_gan = i_pan_gan.transpose((2, 0, 1))
        # image_ms = image_ms.astype(float)
        # image_pan = image_pan.astype(float)
        image_ms = torch.from_numpy(image_ms).type(torch.FloatTensor)
        image_pan = torch.from_numpy(image_pan).type(torch.FloatTensor)
        image_ms_gan = torch.from_numpy(image_ms_gan).type(torch.FloatTensor)
        image_pan_gan = torch.from_numpy(image_pan_gan).type(torch.FloatTensor)
        return image_ms, image_pan, image_ms_gan, image_pan_gan, label, x, y

    def __len__(self):
        return len(self.x)


class dataset_h5(Dataset):
    def __init__(self, dataset):  # 2, 0, 1
        self.ms = to_tensor(dataset['ms'])
        self.pan = to_tensor(dataset['pan'])
        self.label = dataset['label']
        self.xy = dataset['xy']

    def __getitem__(self, item):
        ms_patch = to_tensor(self.ms[item, :, :, :])
        pan_patch = to_tensor(self.pan[item, :, :, :])
        xy = self.xy[item, :]
        label = torch.Tensor(self.label[item])
        label = label.squeeze()
        ms_patch = torch.from_numpy(ms_patch).type(torch.FloatTensor)
        pan_patch = torch.from_numpy(pan_patch).type(torch.FloatTensor)
        return ms_patch, pan_patch, label, xy[0], xy[1]

    def __len__(self):
        return self.ms.shape[0]


class dataset_tri(Dataset):
    def __init__(self, ms, pan, mspan, label, x, y, size):
        self.MS = ms
        self.PAN = pan
        self.MSPAN = mspan
        self.Label = label
        self.x = x
        self.y = y
        self.ms_size = size
        self.pan_size = size * 4

    def __getitem__(self, index):
        ms_size = self.ms_size
        pan_size = self.pan_size
        ms_x = int(self.x[index])
        ms_y = int(self.y[index])
        pan_x = int(4 * ms_x)
        pan_y = int(4 * ms_y)
        image_ms = self.MS[ms_x:ms_x+ms_size, ms_y:ms_y+ms_size, :]
        image_pan = self.PAN[pan_x:pan_x + pan_size, pan_y:pan_y + pan_size]
        image_mspan = self.MSPAN[pan_x:pan_x + pan_size, pan_y:pan_y + pan_size]
        label = torch.Tensor(self.Label[index])
        label = label.squeeze()
        image_ms = image_ms.transpose((2, 0, 1))
        image_pan = np.expand_dims(image_pan, axis=0)
        image_mspan = np.expand_dims(image_mspan, axis=0)
        # image_ms = image_ms.astype(float)
        # image_pan = image_pan.astype(float)
        image_ms = torch.from_numpy(image_ms).type(torch.FloatTensor)
        image_pan = torch.from_numpy(image_pan).type(torch.FloatTensor)
        image_mspan = torch.from_numpy(image_mspan).type(torch.FloatTensor)
        return image_ms, image_pan, image_mspan, label, ms_x, ms_y

    def __len__(self):
        return len(self.x)


class dataset_CT(Dataset):
    def __init__(self, ms, pan, label, x, y, size):
        self.MS = ms
        self.PAN = pan
        self.Label = label
        self.x = x
        self.y = y
        self.ms_size = size
        self.pan_size = size * 4
        self.CT_1 = CT_transform(1)
        self.CT_4 = CT_transform(4)

    def __getitem__(self, index):
        ms_size = self.ms_size
        pan_size = self.pan_size
        ms_x = int(self.x[index])
        ms_y = int(self.y[index])
        pan_x = int(4 * ms_x)
        pan_y = int(4 * ms_y)
        image_ms = self.MS[ms_x:ms_x+ms_size, ms_y:ms_y+ms_size, :]
        image_pan = self.PAN[pan_x:pan_x + pan_size, pan_y:pan_y + pan_size]
        label = torch.Tensor(self.Label[index])
        label = label.squeeze()
        image_ms = image_ms.transpose((2, 0, 1))
        image_pan = np.expand_dims(image_pan, axis=0)
        # image_ms = image_ms.astype(float)
        # image_pan = image_pan.astype(float)
        image_ms = torch.from_numpy(image_ms).type(torch.FloatTensor)
        image_pan = torch.from_numpy(image_pan).type(torch.FloatTensor)
        ms_l, ms_s = self.CT_4.contourlet_decompose(image_ms.unsqueeze(dim=0))
        pan_l1, pan_s1 = self.CT_1.contourlet_decompose(image_pan.unsqueeze(dim=0))
        pan_l2, pan_s2 = self.CT_1.contourlet_decompose(pan_l1)
        pan_l3, pan_s3 = self.CT_1.contourlet_decompose(pan_l2)
        return image_ms, image_pan, \
            (ms_l.squeeze(dim=0), pan_l1.squeeze(dim=0), pan_l2.squeeze(dim=0), pan_l3.squeeze(dim=0)), \
            (ms_s.squeeze(dim=0), pan_s1.squeeze(dim=0), pan_s2.squeeze(dim=0), pan_s3.squeeze(dim=0)), \
            label, ms_x, ms_y

    def __len__(self):
        return len(self.x)


class FusionDataset(Dataset):
    def __init__(self, image, transform=None):
        super().__init__()
        self.image = image
        self.transform = transform

    def __len__(self):
        return len(self.image[0])

    def __getitem__(self, index):
        ms = self.image[0][index]
        pan = self.image[1][index]
        ms = ms.astype(np.float32)
        pan = pan.astype(np.float32)
        if self.transform:
            augmentations = self.transform(image0=ms, image=pan)
            ms = augmentations["image0"]
            pan = augmentations["image"]

        return ms, pan, 0
