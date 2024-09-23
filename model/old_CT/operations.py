import torch
import model.old_CT.sampling as sampling
from torch.nn.functional import conv2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lp_dec(img, h, g, c):
    # img, h, g = img.to(device), h.to(device), g.to(device)
    height, w = img.shape[2], img.shape[3]
    pad_h = int(h.shape[2] / 2)
    padding_per1 = torch.nn.ReflectionPad2d((pad_h, pad_h, pad_h, pad_h))
    low = conv2d(padding_per1(img), h, groups=c)
    low = low[:, :, ::2, ::2]
    high = torch.zeros(img.shape).to(device)
    high[:, :, ::2, ::2] = low
    pad_g = int(g.shape[2] / 2)
    padding_per2 = torch.nn.ReflectionPad2d((pad_g, pad_g, pad_g, pad_g))
    high = conv2d(padding_per2(high), g, groups=c)
    high = img - high
    return low, high


def lp_rec(low_band, high, h, g, c):
    high_ = high
    pad_h = int(h.shape[2] / 2)
    padding_per = torch.nn.ReflectionPad2d((pad_h, pad_h, pad_h, pad_h))
    high = conv2d(padding_per(high), h, padding=0, groups=c)
    high = high[:, :, ::2, ::2]
    high = low_band - high

    img = torch.zeros((low_band.shape[0], low_band.shape[1],
                       low_band.shape[2]*2, low_band.shape[3]*2)).to(device)
    img[:, :, ::2, ::2] = high
    pad_g = int(g.shape[2] / 2)
    padding_per = torch.nn.ReflectionPad2d((pad_g, pad_g, pad_g, pad_g))
    img = conv2d(padding_per(img), g, padding=0, groups=c)
    img = img + high_
    return img


def dfb_dec(img, h0, h1, c, index='', name=None):

    h, w = img.shape[2], img.shape[3]
    if name == 'haar':
        padding0 = (0, 1)
        padding1 = (0, 1)
    else:
        pass

    padding_per_2 = torch.nn.ReflectionPad2d((2, 2, 2, 2))

    y0 = sampling.q_sampling(conv2d(padding_per_2(img), h0, padding=0, groups=c), q_mode='q0', op_mode='down')
    y1 = sampling.q_sampling(conv2d(padding_per_2(img), h1, padding=0, groups=c), q_mode='q0', op_mode='down')
    # save_image(y0, "test_image/y0_band_"+index+".png")
    # save_image(y1, "test_image/y1_band_"+index+".png")
    # print(y0.shape, y1.shape)
    # y0 = y0.to(device)
    # y1 = y1.to(device)
    y00 = sampling.q_sampling(conv2d(padding_per_2(y0), h0, padding=0, groups=c), q_mode='q1', op_mode='down')
    y01 = sampling.q_sampling(conv2d(padding_per_2(y0), h1, padding=0, groups=c), q_mode='q1', op_mode='down')
    y10 = sampling.q_sampling(conv2d(padding_per_2(y1), h0, padding=0, groups=c), q_mode='q1', op_mode='down')
    y11 = sampling.q_sampling(conv2d(padding_per_2(y1), h1, padding=0, groups=c), q_mode='q1', op_mode='down')
    # save_image(y00[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4], "test_image/sub_band"+index+"_00.png")
    # save_image(y01[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4], "test_image/sub_band"+index+"_01.png")
    # save_image(y10[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4], "test_image/sub_band"+index+"_10.png")
    # save_image(y11[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4], "test_image/sub_band"+index+"_11.png")
    # print(y00.shape, y01.shape, y10.shape, y11.shape)
    return torch.cat((y00, y01, y10, y11), dim=1)[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4]


def dfb_rec(sub_bands, g0, g1, c, name=None):
    h, w = sub_bands.shape[2], sub_bands.shape[3]
    pad = torch.nn.ReflectionPad2d((w // 2, w // 2, h // 2, h // 2))
    sub_bands = pad(sub_bands)

    padding_per = torch.nn.ReflectionPad2d((2, 2, 2, 2))
    # padding_per = torch.nn.ZeroPad2d((2,2,2,2))

    # print('pad: ', pad)
    if name == 'haar':
        padding0 = (0, 1)
        padding1 = (0, 1)
    else:
        # padding_per_1 = torch.nn.ReflectionPad2d((2,2,2,2))
        # padding_per_3 = torch.nn.ReflectionPad2d((2,2,2,2))
        padding_per_1 = torch.nn.ReflectionPad2d((1, 1, 1, 1))
        padding_per_3 = torch.nn.ReflectionPad2d((3, 3, 3, 3))

    y00 = sampling.q_sampling(sub_bands[:, 0:c], q_mode='q1', op_mode='up')
    y01 = sampling.q_sampling(sub_bands[:, c:2 * c], q_mode='q1', op_mode='up')
    # y00 = y00.to(device)
    # y01 = y01.to(device)
    y00 = conv2d(padding_per_1(y00), g0, padding=0, groups=c)
    y01 = conv2d(padding_per_3(y01), g1, padding=0, groups=c)
    y0 = y00 + y01

    y10 = sampling.q_sampling(sub_bands[:, 2 * c:3 * c], q_mode='q1', op_mode='up')
    y11 = sampling.q_sampling(sub_bands[:, 3 * c:4 * c], q_mode='q1', op_mode='up')
    # y10 = y10.to(device)
    # y11 = y11.to(device)
    y10 = conv2d(padding_per_1(y10), g0, padding=0, groups=c)
    y11 = conv2d(padding_per_3(y11), g1, padding=0, groups=c)
    y1 = y10 + y11

    y0 = sampling.q_sampling(y0, q_mode='q0', op_mode='up')
    y1 = sampling.q_sampling(y1, q_mode='q0', op_mode='up')
    # y0 = y0.to(device)
    # y1 = y1.to(device)
    y0 = conv2d(padding_per_1(y0), g0, padding=0, groups=c)
    y1 = conv2d(padding_per_3(y1), g1, padding=0, groups=c)
    # print('y0 y1 shape', y0.shape, y1.shape)
    return y0 + y1
