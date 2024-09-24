import numpy as np
import torch
import utils.config as config
from PIL import Image
from tqdm import tqdm
import os

DEVICE = config.DEVICE
colormap = config.COLORMAP
size = config.SIZE


def verify(model, loader1, loader2, expo, num, mode='2'):
    if os.path.exists(expo) == 0:
        os.makedirs(expo)
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    label_np2 = np.zeros([size[0], size[1]])
    loop = tqdm(loader1, leave=True)
    with torch.no_grad():
        if mode == '1':
            for batch_idx, (data, target, x, y) in enumerate(loop):
                data, target = data.to(DEVICE), target.to(DEVICE)
                out = model(data, mode=mode)
                pred = out.data.max(1, keepdim=True)[1]
                for i in range(int(data.shape[0])):
                    label_np1[int(x[i])][int(y[i])] = int(pred[i])
                    label_np2[int(x[i])][int(y[i])] = int(pred[i])
                loop.set_postfix(mode='verify')
            loop.close()

            loop = tqdm(loader2, leave=True)
            with torch.no_grad():
                for batch_idx, (data, target, x, y) in enumerate(loop):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    out = model(data, mode=mode)
                    pred = out.data.max(1, keepdim=True)[1]
                    for i in range(int(data.shape[0])):
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    loop.set_postfix(mode='verify')
            loop.close()
        elif mode == '2':
            for batch_idx, (data1, data2, target, x, y) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                out = model(data1, data2)
                pred = out.data.max(1, keepdim=True)[1]
                for i in range(int(data1.shape[0])):
                    label_np1[int(x[i])][int(y[i])] = int(pred[i])
                    label_np2[int(x[i])][int(y[i])] = int(pred[i])
                loop.set_postfix(mode='verify')
            loop.close()

            loop = tqdm(loader2, leave=True)
            with torch.no_grad():
                for batch_idx, (data1, data2, target, x, y) in enumerate(loop):
                    data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                    out = model(data1, data2)
                    pred = out.data.max(1, keepdim=True)[1]
                    for i in range(int(data1.shape[0])):
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    loop.set_postfix(mode='verify')
        elif mode == '3':
            for batch_idx, (data1, data2, data3, target, x, y) in enumerate(loop):
                data1, data2, data3, target = data1.to(DEVICE), \
                                              data2.to(DEVICE), \
                                              data3.to(DEVICE), \
                                              target.to(DEVICE)
                out = model(data1, data2, data3)
                pred = out.data.max(1, keepdim=True)[1]
                for i in range(int(data1.shape[0])):
                    label_np1[int(x[i])][int(y[i])] = int(pred[i])
                    label_np2[int(x[i])][int(y[i])] = int(pred[i])
                loop.set_postfix(mode='verify')
            loop.close()

            loop = tqdm(loader2, leave=True)
            with torch.no_grad():
                for batch_idx, (data1, data2, data3, target, x, y) in enumerate(loop):
                    data1, data2, data3, target = data1.to(DEVICE), \
                                                  data2.to(DEVICE), \
                                                  data3.to(DEVICE), \
                                                  target.to(DEVICE)
                    out = model(data1, data2, data3)
                    pred = out.data.max(1, keepdim=True)[1]
                    for i in range(int(data1.shape[0])):
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    loop.set_postfix(mode='verify')
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture1 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_1.jpg"
    picture1.save(savepath)

    for i in range(label_np2.shape[0]):
        for j in range(label_np2.shape[1]):
            label_pic[i][j] = colormap[int(label_np2[i][j])]
    picture2 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_2.jpg"
    picture2.save(savepath)


def verify_second(model, loader1, loader2, expo, inpo, num, mode='2'):
    if os.path.exists(expo) == 0:
        os.makedirs(expo)
    model.load_state_dict(torch.load(inpo))
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    label_np2 = np.zeros([size[0], size[1]])
    loop = tqdm(loader1, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, target, x, y) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            out = model(data1, data2)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(data1.shape[0])):
                label_np1[int(x[i])][int(y[i])] = int(pred[i])
                label_np2[int(x[i])][int(y[i])] = int(pred[i])
            loop.set_postfix(mode='verify')
        loop.close()
        loop = tqdm(loader2, leave=True)
        with torch.no_grad():
            for batch_idx, (data1, data2, target, x, y) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                out = model(data1, data2)
                pred = out.data.max(1, keepdim=True)[1]
                for i in range(int(data1.shape[0])):
                    label_np2[int(x[i])][int(y[i])] = int(pred[i])
                loop.set_postfix(mode='verify')
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture1 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_1.jpg"
    picture1.save(savepath)

    for i in range(label_np2.shape[0]):
        for j in range(label_np2.shape[1]):
            label_pic[i][j] = colormap[int(label_np2[i][j])]
    picture2 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_2.jpg"
    picture2.save(savepath)


def verify_CT(model, loader1, loader2, expo, num, mode='2'):
    if os.path.exists(expo) == 0:
        os.makedirs(expo)
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    label_np2 = np.zeros([size[0], size[1]])
    loop = tqdm(loader1, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, target, x, y) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            out = model(data1, data2)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(data1.shape[0])):
                label_np1[int(x[i])][int(y[i])] = int(pred[i])
                label_np2[int(x[i])][int(y[i])] = int(pred[i])
            loop.set_postfix(mode='verify')
        loop.close()

        loop = tqdm(loader2, leave=True)
        with torch.no_grad():
            for batch_idx, (data1, data2, data3, data4, target, x, y) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                out = model(data1, data2)
                pred = out.data.max(1, keepdim=True)[1]
                for i in range(int(data1.shape[0])):
                    label_np2[int(x[i])][int(y[i])] = int(pred[i])
                loop.set_postfix(mode='verify')
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture1 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_1.jpg"
    picture1.save(savepath)

    for i in range(label_np2.shape[0]):
        for j in range(label_np2.shape[1]):
            label_pic[i][j] = colormap[int(label_np2[i][j])]
    picture2 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = expo + str(num) + "_pic_2.jpg"
    picture2.save(savepath)


def verify_SCLN(model, verify_loader1, verify_loader2):
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    loop = tqdm(verify_loader1, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, target, x, y) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            out = model(data1)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(data1.shape[0])):
                label_np1[int(x[i])][int(y[i])] = int(pred[i])
            # if batch_idx % int(len(verify_loader1) / 10) == 0:
            #     print('Verify rate [{}/{} ({:.0f}%)]'.format(
            #         batch_idx * len(target), len(verify_loader1.dataset),
            #         100. * batch_idx / len(verify_loader1)))
            loop.set_postfix(mode='verify')
    loop.close()
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    picture.save(config.data_city + "_picture_label" + "_1.jpg")
    loop = tqdm(verify_loader2, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, _, _, target, x, y) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            out = model(data1)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(data1.shape[0])):
                label_np1[int(x[i])][int(y[i])] = int(pred[i])
            # if batch_idx % int(len(verify_loader2) / 10) == 0:
            #     print('Verify rate [{}/{} ({:.0f}%)]'.format(
            #         batch_idx * len(target), len(verify_loader2.dataset),
            #         100. * batch_idx / len(verify_loader2)))
            loop.set_postfix(mode='verify')
    loop.close()
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    picture.save(config.data_city + "picture_label_" + "_2.jpg")