from libtiff import TIFF
import numpy as np
import cv2, h5py, torch, sys, math, os
import scipy.io
from image_convert.IHS import pan2ms
# import config
# size = config.DATA_DICT[config.DATA_CITY]['size']


# 用于将标签mat文件转化成np文件
def label_mat2np(cfg):
    path = cfg['data_address']
    label_np_float = h5py.File(path+'label.mat')  # 创建引入库并创建h5文件
    label_np_float = label_np_float['label']
    label_np = np.array(label_np_float, dtype='uint8')
    print(label_np.shape)
    np.save(path + 'label.npy', np.transpose(label_np))


def colorlist(args):
    a = np.load("label9.npy")
    q = []
    for i in range(args.Categories):
        z = np.argwhere(a == i)
        q.append(z)
    pic = cv2.imread('GT.png')
    for i in range(args.Categories):
        for j in range(100):
            if pic[q[i][j][0]][q[i][j][1]].sum() != 0:
                print(pic[q[i][j][0]][q[i][j][1]], i)
                break


def read_tif(cfg, mode):
    if mode == 'ms':
        filename = cfg['data_address'] + 'ms4.tif'
    elif mode == 'pan':
        filename = cfg['data_address'] + 'pan.tif'
    else:
        raise ValueError("mode")
    tif = TIFF.open(filename, mode='r')
    image = tif.read_image()
    return image


def read_h5(filename):
    file = h5py.File(filename, 'r')
    dict = {}
    for key in file.keys():
        array = np.array(file[key])
        dict[key] = array
    file.close()
    return dict


def dataset_cut(pan, ms, matrix, valid, cfg, mode):
    if mode == 'train':
        filename = cfg['data_address']+str(cfg['patch_size'])+"_train.h5"
    elif mode == 'color':
        filename = cfg['data_address']+str(cfg['patch_size'])+"_color.h5"
    else:
        raise ValueError("mode")
    patch_size = cfg['patch_size']
    pan = np.expand_dims(pan, axis=2)
    pan_result = []
    ms_result = []
    label_result = []
    xy_result = []
    for item in valid:
        x, y = int(matrix[0][item]), int(matrix[1][item])
        pan_result.append(pan[x:x + patch_size*4, y:y + patch_size*4, :].transpose((2, 0, 1)))
        ms_result.append(ms[x:x + patch_size, y:y + patch_size, :].transpose((2, 0, 1)))
        label_result.append(matrix[2][item])
        xy_result.append([int(matrix[0][item]), int(matrix[1][item])])
    # print(round(sys.getsizeof(pan_result) / 1024 / 1024, 2))
    # pan_result = np.array(pan_result)
    # ms_result = np.array(ms_result)
    label_result = np.array(label_result)
    xy_result = np.array(xy_result)
    save_h5(filename, pan_result, ms_result, label_result, xy_result)
    return pan_result, ms_result


def save_h5(filename, pan, ms, label, xy):
    f = h5py.File(filename, 'w')  # 写入文件
    f['pan'] = pan  # 名称为image
    f['ms'] = ms
    f['label'] = label
    f['xy'] = xy
    f.close()  # 关闭文件


def read_tfw(file_name):
    with open(file_name, 'r') as f:
        tfw_contents = f.read()
        print(tfw_contents)


def data_padding(array, cfg, mode):  # 填充数据以方便切割
    axis = len(array.shape)
    patch_size = cfg['patch_size'] if axis ==3 else cfg['patch_size']*4
    print("*********************")
    array = to_tensor(array)
    Interpolation = cv2.BORDER_REFLECT_101  # 填充方式为对称
    # top_size, bottom_size, left_size, right_size = (int(patch_size / 2) - 1, int(patch_size / 2),
    #                                                 int(patch_size / 2) - 1, int(patch_size / 2))
    top_size, bottom_size, left_size, right_size = (0, patch_size-1,
                                                    0, patch_size-1)
    array = cv2.copyMakeBorder(array, top_size, bottom_size,
                            left_size, right_size, Interpolation)
    if axis == 3:
        array_row, array_column, array_channel = np.shape(array)
        print("数据行{}.数据列{}.图片通道{}".format(array_row, array_column, array_channel))
    elif axis == 2:
        array_row, array_column= np.shape(array)
        print("数据行{}.数据列{}.图片通道{}".format(array_row, array_column, 1))
    return array


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def data_show(matrix):
    label_element, element_count = np.unique(matrix, return_counts=True)  # 去除重复数据并进行排序后输出
    print('************************')
    Categories_Number = len(label_element) - 1  # Categories_Number = 11
    label_row, label_column = np.shape(matrix)  # 2001, 2101
    print("标签类别{}.每一类的数量{}.标签矩阵的行{}.标签矩阵的列{}.有真实标签的类别数量{}"
          .format(label_element, element_count, label_row, label_column, Categories_Number))


def pan_pic_product(image, index, args):
    temp = 0
    pic_size = args['dqtl']['pic_size']
    shape = [4, index[0]*pic_size, index[1]*pic_size]
    result = np.zeros(shape)
    for i in range(index[0]):
        for j in range(index[1]):
            result[:, i * pic_size:(i+1)*pic_size, j * pic_size:(j+1)*pic_size] = image[temp]
            temp += 1
    size = args['DATA_DICT'][args['data_city']]['size']
    return result[:, 0:size[0], 0:size[1]]


def split_data_old(label, cfg):
    size = cfg['DATA_DICT'][cfg['data_city']]['size']
    # 处理数据集
    the_matrix = [[] for i in range(3)]  # the_matrix[0],the_matrix[1]用来存储(x,y),the_matrix[2]用来存储label
    matrix_ = [[] for i in range(2)]  # matrix_用来存储对应标签0或1的索引
    for i in range(3):
        the_matrix[i] = np.zeros(shape=(size[0] * size[1], 1))
    temp = 0
    for i in range(size[0]):
        for j in range(size[1]):
            the_matrix[0][temp] = i
            the_matrix[1][temp] = j
            the_matrix[2][temp] = label[i][j]
            if label[i][j] == 0:
                matrix_[0].append(temp)
            else:
                matrix_[1].append(temp)
            temp += 1
    for i in range(2):
        print("标签为{}的标签集合的大小为{}".format(i, len(matrix_[i])))
    return the_matrix, matrix_


def split_data(train_label, test_label, label, cfg):
    size = cfg['DATA_DICT'][cfg['data_city']]['size']
    # 处理数据集
    the_matrix = [[] for i in range(3)]  # the_matrix[0],the_matrix[1]用来存储(x,y),the_matrix[2]用来存储label
    matrix_ = [[] for i in range(3)]  # matrix_用来存储未被标记的和train和test的标记区域
    for i in range(3):
        the_matrix[i] = np.zeros(shape=(size[0] * size[1], 1))
    temp = 0
    for i in range(size[0]):
        for j in range(size[1]):
            the_matrix[0][temp] = i
            the_matrix[1][temp] = j
            the_matrix[2][temp] = label[i][j]
            if train_label[i][j] != 0:
                matrix_[1].append(temp)
            elif test_label[i][j] != 0:
                matrix_[2].append(temp)
            else:
                matrix_[0].append(temp)
            temp += 1
    for i in range(len(matrix_)):
        print("标签为{}的标签集合的大小为{}".format(i, len(matrix_[i])))
    return the_matrix, matrix_


def data_process_dqtl_new(m, p, cfg):
    return 0

def data_process_dqtl_stage1(m, p, cfg):
    size = cfg['DATA_DICT'][cfg['data_city']]['size']
    pic_size = cfg['dqtl']['pic_size']
    index_x, index_y = math.ceil(size[0]/pic_size), \
                       math.ceil(size[1]/pic_size)
    m = cv2.copyMakeBorder(m, 0, index_x*pic_size-size[0], 0, index_y*pic_size-size[1], cv2.BORDER_REFLECT_101)  # (2048, 2304, 4) 8, 9
    m = m.transpose((2, 0, 1))
    m = to_tensor(m)
    if os.path.exists(cfg['data_address'] + '/pan.npy'):
        p = np.load(cfg['data_address'] + '/pan.npy')
    else:
        p = pan2ms(p, size)
        np.save(cfg['data_address'] + '/pan.npy', p)
    p = cv2.copyMakeBorder(p, 0, index_x*pic_size-size[0], 0, index_y*pic_size-size[1], cv2.BORDER_REFLECT_101)  # (2048, 2304, 4) 9, 9
    p = p.transpose((2, 0, 1))
    p = to_tensor(p)
    the_matrix = [[] for i in range(2)]
    batch_size = pic_size
    temp = 0
    for i in range(index_x):
        for j in range(index_y):
            the_matrix[0].append(m[:, i * batch_size:(i + 1) * batch_size, j * batch_size:(j + 1) * batch_size])
            temp += 1
    temp = 0
    for i in range(index_x):
        for j in range(index_y):
            the_matrix[1].append(p[:, i * batch_size:(i + 1) * batch_size, j * batch_size:(j + 1) * batch_size])
            temp += 1
    return the_matrix, index_x, index_y


def equalize_histogram(band):
    hist, bins = np.histogram(band.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[band]


def visualize_img(img, name, equalize=1):
    from PIL import Image
    img = img[0].cpu().detach().numpy()

    if img.shape[0] == 4:
        band_data = img[(2, 1, 0), :, :]
        scaled_data = []
        for i, band in enumerate(band_data):
            band_min, band_max = band.min(), band.max()
            scaled_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            if equalize:
                scaled_band = equalize_histogram(scaled_band)
            # scaled_band = adjust_contrast(scaled_band, 0.95)
            # scaled_band = adjust_exposure(scaled_band, 0.95)
            scaled_data.append(scaled_band)

        processed_array = np.dstack(scaled_data)
        # processed_array = adjust_brightness_hsv(processed_array, 0.85)
    elif img.shape[0] == 1:
        band = img[0]
        band_min, band_max = band.min(), band.max()
        processed_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
        if equalize:
            processed_band = equalize_histogram(processed_band)
        processed_array = processed_band

    else:
        raise ValueError("Unsupported image type. Please use 'multispectral' or 'pan'.")

    result = Image.fromarray(processed_array, 'RGB' if img.shape[0] == 4 else 'L')
    result.show()
    result.save(name)


def visualize_channels(tensor, num_channels=8, cols=4, name=''):

    """
    可视化指定数量的通道。
    :param tensor: BCHW 形状的张量。
    :param num_channels: 要展示的通道数量。
    :param cols: 每行显示的图像数量。
    """
    import matplotlib.pyplot as plt
    tensor = tensor[0]  # 选择批次中的第一个样本
    channels = tensor.shape[0]  # 获取通道数

    # 如果通道数为1，仅展示这一个通道
    if channels == 1:
        plt.imshow(tensor[0].cpu().detach().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title('Single Channel_'+name)
        plt.savefig('Single Channel_'+name)
        plt.show()

        return

    rows = num_channels // cols + int(num_channels % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(tensor[i].cpu().detach().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i + 1}-'+name)

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'Channel {i + 1}-'+name)
    plt.show()
