# from function.function import read_tif
import numpy as np
import random

# 下采样，通过平均值进行下采样
def unsampling(im, scale):
    original_height, original_width = im.shape
    resized_image = np.zeros([int(original_height/scale), int(original_width/scale)])
    for i in range(0, original_height, scale):
        for j in range(0, original_width, scale):
            resized_image[int(i/scale), int(j/scale)] = np.mean(im[i:i + scale, j:j + scale], axis=(0, 1))
    return resized_image

def pan2ms(pan, size):
    p = unsampling(pan, 2)
    result = np.zeros(size)
    for i in range(size[2]):
        result[:, :, i] = p[i%2::2, int(i/2)::2]
    return result

# 下池化，即将图像放大time倍，周围区域取零
def unpooling(pic, time):
    pic_unpooling = np.zeros([pic.shape[0]*time, pic.shape[1]*time, pic.shape[2]])
    for i in range(pic.shape[2]):
        for j in range(pic.shape[0]):
            for k in range(pic.shape[1]):
                m, n = random.randint(0, time-1), random.randint(0, time-1)
                pic_unpooling[time*j+m, time*k+n, i] = pic[j, k, i]
    return pic_unpooling

# 将单通道图像复制到n通道
def raw_3copy(image_raw, n):
    # image = np.expand_dims(image_raw, axis=2)
    # image = np.concatenate((image_raw, image_raw, image_raw), axis=-1)
    image_raw = image_raw[:, :, np.newaxis]
    image = image_raw.repeat([n], axis=2)
    return image


def IHS_tran(MS, PAN):
    MS_unpooling = unpooling(MS, MS.shape[2])
    for i in range(MS.shape[2]):
        if i == 0:
            I = MS_unpooling[:, :, i]
        else:
            I = (I*i + MS_unpooling[:, :, i])/(i+1)
    delta = PAN - I
    result = MS_unpooling + raw_3copy(delta, MS.shape[2])
    for i in range(result.shape[2]):
        if i == 0:
            MSPAN = result[:,:,i]
        else:
            MSPAN = (MSPAN*i + result[:,:,i])/(i+1)
    return MSPAN


# if __name__ == '__main__':
#     MS = read_tif(config.DATA_DICT[config.data_city]['ms'])
#     PAN = read_tif(config.DATA_DICT[config.data_city]['pan'])
#     # print(IHS_tran(MS, PAN).shape)
#     print(pan2ms(PAN, config.DATA_DICT[config.data_city]['size']).shape)