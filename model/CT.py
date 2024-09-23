from model.old_CT import filters, operations
import torch
import time
import numpy as np


def contourlet_transform(img, name='thanh'):
    # img = img.to(device)
    low_band, sub_bands = contourlet_decompose(img, name=name)
    # print(low_band.shape, sub_bands.shape)
    img_ = contourlet_recompose(low_band, sub_bands, name=name)
    return img_


def contourlet_decompose(img, index='', name='thanh'):
    img = img.to(device)
    channel = img.shape[1]
    # 9-7 filters
    h, g = filters.lp_filters(channel)
    # h, g = filters.lp_filters(channel)
    # Laplacian Pyramid decompose
    low_band, high = operations.lp_dec(img, h, g, channel)
    # save_image(low_band, 'test_image/low_band'+index+'.png')
    # save_image(high, 'test_image/high_band'+index+'.png')
    # DFB filters
    h0, h1 = filters.dfb_filters(channel, mode='d', name=name)
    # DFB decompose
    sub_bands = operations.dfb_dec(high, h0, h1, channel, index, name=name)
    return low_band, sub_bands


def contourlet_recompose(low_band, sub_bands, name='thanh'):
    channel = low_band.shape[1]
    # DFB filters
    g0, g1 = filters.dfb_filters(channel, mode='r', name=name)
    # DFB recompose
    high = operations.dfb_rec(sub_bands, g0, g1, channel, name=name)
    # 9-7 filters
    h, g = filters.lp_filters(channel)
    # Laplacian recompose
    img = operations.lp_rec(low_band, high, h, g, channel)
    return img


def contourlet_only_LP(img):
    channel = img.shape[1]
    # 9-7 filters
    h, g = filters.lp_filters(channel)
    # Laplacian Pyramid decompose
    low_band, high = operations.lp_dec(img, h, g, channel)
    img = operations.lp_rec(low_band, high, h, g, channel)
    return img


def contourlet_only_DFB(img):
    channel = img.shape[1]
    h0, h1 = filters.dfb_filters(channel, mode='d', name='thanh')
    g0, g1 = filters.dfb_filters(channel, mode='r', name='thanh')
    sub_bands = operations.dfb_dec(img, h0, h1, channel, name='thanh')
    img = operations.dfb_rec(sub_bands, g0, g1, channel, name='thanh')
    return img


def decompose_level(img, level):
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    img_l, img_s = [], []
    for i in range(level):
        low_band, sub_band = contourlet_decompose(img)
        img_l.append(np.array(low_band[0].cpu()).transpose((1, 2, 0)))
        print(low_band[0].shape)
        img_s.append(np.array(sub_band[0].cpu()).transpose((1, 2, 0)))
        img = low_band
    return img_l, img_s


def recompose_level(low_band, sub_band, level):
    for i in range(level):
        img = contourlet_recompose(low_band[i], sub_band[i])
        print(img.shape)




def test_ct():
    img = torch.randn([1, 3200, 3320])
    low, high = decompose_level(img, 3)
    # recompose_level(low, high, 3)


def test():
    x = torch.randn([20, 1, 64, 64]).to(device)
    contourlet_decompose(x)
    contourlet_decompose(x)
    contourlet_decompose(x)
    contourlet_decompose(x)
    contourlet_decompose(x)


def test1():
    import cv2
    import numpy as np
    import os
    img = cv2.imread("contourlet/image/zone.png")
    img = np.transpose(img/255.0, (2, 0, 1)).astype(np.float32)

    # img = read_tif(config.DATA_ADDRESS+'pan.tif')
    # img = np.expand_dims(img[:1600, :1600], axis=2)
    # img = to_tensor(img).transpose((2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img).to(device)
    FILE = '../test_image/'
    if os.path.exists(FILE) == 0:
        os.makedirs(FILE)
    print(img.shape)
    time1 = time.time()
    for i in range(5):
        low, sub = contourlet_decompose(img, str(i))
        img = low
    time2 = time.time()
    print(time2-time1)
    # for i in range(5):
    #     low, sub = contourlet_decompose(img, str(i))
    #     save_image(low, FILE+"low"+str(i)+"_band.png")
    #     img = low
    #     for j in range(low.shape[1]):
    #         save_image(low[0][j], FILE+"low"+str(i)+"_band" + str(j) + ".png")
    #     for j in range(sub.shape[1]):
    #         save_image(sub[0][j], FILE+"sub"+str(i)+"_band" + str(j) + ".png")


def test2():
    import cv2
    import numpy as np
    import os
    img = cv2.imread("contourlet/image/zoneplate.png")
    img = np.transpose(img / 255.0, (2, 0, 1)).astype(np.float32)

    # img = read_tif(config.DATA_ADDRESS+'pan.tif')
    # img = np.expand_dims(img[:1600, :1600], axis=2)
    # img = to_tensor(img).transpose((2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img).to(device)
    FILE = '../test_image/'
    if os.path.exists(FILE) == 0:
        os.makedirs(FILE)
    print(img.shape)
    h0, h1 = filters.dfb_filters(img.shape[1], mode='d', name='thanh')
    sub_bands = operations.dfb_dec(img, h0.to(device), h1.to(device), img.shape[1], name='thanh')


if __name__ == "__main__":
    test()
