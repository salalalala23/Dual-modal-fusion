import utils.config as config
# from DQTL import config1
from tqdm import tqdm
from train.loss_function import KL_loss
import random
from torchvision.utils import save_image
import torch
import torch.nn as nn
import numpy as np
import os
DEVICE = config.DEVICE


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(model, train_loader, optimizer, epoch, time, mode='2'):
    loop = tqdm(train_loader, leave=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if mode == '1':
        for batch_idx, (data, target, _, _) in enumerate(loop):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data, mode=mode)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss, epoch=epoch, time=time, mode='train')
    elif mode == '2':
        for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss, epoch=epoch, time=time, mode='train')
    elif mode == '3':
        for batch_idx, (data1, data2, data3, target, _, _) in enumerate(loop):
            data1, data2, data3, target = data1.to(DEVICE), data2.to(DEVICE), data3.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data1, data2, data3)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss, epoch=epoch, time=time, mode='train')
    loop.close()


def train_second(model, train_loader, valid_loader, optimizer, epoch, expo, time):
    criterion = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    best_loss = float('inf')
    for ep in range(epoch):
        loop = tqdm(train_loader, leave=True)
        model.train()
        for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(), epoch=ep, time=time, mode='train')
        loop.close()
        loop = tqdm(valid_loader, leave=True)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                output = model(data1, data2)
                loss = criterion(output, target.long())
                val_loss += loss.item() * data1.size(0)
                loop.set_postfix(loss=val_loss, best_loss=best_loss, epoch=ep, time=time, mode='valid')
                if val_loss > best_loss:
                    break
            loop.close()
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            torch.save(best_weights, expo)


def train_CT(model, train_loader, valid_loader, optimizer, epoch, expo, time):
    criterion = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    best_loss = float('inf')
    for ep in range(epoch):
        loop = tqdm(train_loader, leave=True)
        model.train()
        for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(), best_loss=best_loss, epoch=ep, time=time, mode='train')
        loop.close()
        loop = tqdm(valid_loader, leave=True)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                output = model(data1, data2)
                loss = criterion(output, target.long())
                val_loss += loss.item() * data1.size(0)
                if val_loss > best_loss:
                    break
                loop.set_postfix(loss=val_loss, epoch=ep, time=time, mode='valid')
            loop.close()
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            torch.save(best_weights, expo)


def train_SCLN(model, train_loader, triplet_loss, optimizer, epoch):
    # setup_seed(3407)
    loop = tqdm(train_loader, leave=True)
    model.train()
    loss_list = []
    IOU_list = []
    for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(loop):
        data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
        data3, data4 = data3.to(DEVICE), data4.to(DEVICE)
        data = torch.concat([data1, data2, data3, data4])
        bs = len(data1)
        optimizer.zero_grad()
        out = model(data)
        # loss1 = criterion(out[:bs], target[:bs].long())
        # loss2 = criterion(out[bs:2*bs], target[:bs].long())
        # loss3 = criterion(out[2*bs:3*bs], target[2*bs:3*bs].long())
        # loss4 = criterion(out[3 * bs:], target[3 * bs:].long())
        # loss2 = triplet_loss(out[:bs], out[bs:2 * bs], out[2 * bs:3 * bs])
        # loss3 = triplet_loss(out[bs:2 * bs], out[:bs], out[-bs:])
        loss = KL_loss(out[:bs], out[bs:2 * bs], out[2*bs:3*bs], out[3*bs:], target)
        loss_list.append(loss)
        # IOU_list.append(IOU)
        loss.backward()
        optimizer.step()
        # if batch_idx % int(len(train_loader) / 10) == 0:
        #     print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #         epoch, batch_idx * len(data1), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        loop.set_postfix(loss=loss, epoch=epoch+1, mode='train')
    # for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(loop):
    #     data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
    #     data3, data4 = data3.to(DEVICE), data4.to(DEVICE)
    #     bs = len(data1)
    #     data_origin = torch.cat([data1, data2], dim=0)
    #     data_generate = torch.cat([data3, data4], dim=0)
    #     target = torch.cat((target, target))
    #     optimizer.zero_grad()
    #     out = model(data_origin, data_generate)
    #     loss = criterion(out[:2*bs], target.long())
    #     # loss = DL_loss(out[:bs], out[bs:2 * bs], out[2*bs:3*bs], out[3*bs:], loss1, loss1)
    #     # loss = loss1 + loss
    #     loss.backward()
    #     optimizer.step()
    #     # if batch_idx % int(len(train_loader) / 10) == 0:
    #     #     print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
    #     #         epoch, batch_idx * len(data1), len(train_loader.dataset),
    #     #         100. * batch_idx / len(train_loader), loss.item()))
    #     loop.set_postfix(loss=loss, epoch=epoch, mode='train')


def train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (pan, ms) in enumerate(loop):
        pan = pan.to(config.DEVICE)
        ms = ms.to(config.DEVICE)

        # 训练判别器 H Z
        with torch.cuda.amp.autocast():
            fake_pan = gen_H(ms)
            D_H_real = disc_H(pan)
            D_H_fake = disc_H(fake_pan.detach())
            D_H_real_loss = MSE(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = MSE(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_ms = gen_Z(pan)
            D_Z_real = disc_Z(ms)
            D_Z_fake = disc_Z(fake_ms.detach())
            D_Z_real_loss = MSE(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练生成器 H Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_pan)
            D_Z_fake = disc_Z(fake_ms)
            loss_G_H = MSE(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = MSE(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_ms = gen_Z(fake_pan)
            cycle_pan = gen_H(fake_ms)
            cycle_ms_loss = L1(ms, cycle_ms)
            cycle_pan_loss = L1(pan, cycle_pan)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # 防止生成一个一样的？？
            identity_ms = gen_Z(ms)
            identity_pan = gen_H(pan)
            identity_ms_loss = L1(ms, identity_ms)
            identity_pan_loss = L1(pan, identity_pan)

            G_loss = (   # 6个loss
                loss_G_H
                + loss_G_Z
                + cycle_ms_loss * config1.LAMBDA_CYCLE
                + cycle_pan_loss * config1.LAMBDA_CYCLE
                + identity_ms_loss * config1.LAMBDA_IDENTITY
                + identity_pan_loss * config1.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        if os.path.exists(config.data_city+"_"+str(config1.PIC_SIZE)+"_saved_images1") == 0:
            os.makedirs(config.data_city+"_"+str(config1.PIC_SIZE)+"_saved_images1")
        if idx % 1 == 0:
            save_image(fake_pan, config.data_city+"_"+str(config1.PIC_SIZE)+f"_saved_images1/fake_pan_{idx}.png")
            save_image(fake_ms, config.data_city+"_"+str(config1.PIC_SIZE)+f"_saved_images1/fake_ms_{idx}.png")
            save_image(pan, config.data_city+"_"+str(config1.PIC_SIZE)+f"_saved_images1/pan_{idx}.png")
            save_image(ms, config.data_city+"_"+str(config1.PIC_SIZE)+f"_saved_images1/ms_{idx}.png")

        fake_pan = fake_pan.cpu().detach().numpy()
        fake_ms = fake_ms.cpu().detach().numpy()
        if idx == 0:
            test_ms = fake_pan
            test_pan = fake_ms
        else:
            test_ms = np.append(test_ms, fake_pan, axis=0)
            test_pan = np.append(test_pan, fake_ms, axis=0)

        loop.set_postfix(H_real=D_H_real_loss/(idx+1), H_fake=D_H_fake_loss/(idx+1), epoch=epoch)
    return test_ms, test_pan


def train_gn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (pan, ms) in enumerate(loop):
        pan = pan.to(config.DEVICE)
        ms = ms.to(config.DEVICE)
        noise_ms = torch.randn([pan.shape[0], 4, config1.PIC_SIZE, config1.PIC_SIZE]).to(config.DEVICE)
        noise_pan = torch.randn([ms.shape[0], 4, config1.PIC_SIZE, config1.PIC_SIZE]).to(config.DEVICE)
        # 训练判别器 H Z
        with torch.cuda.amp.autocast():
            fake_pan = gen_H(noise_pan)
            D_H_real = disc_H(pan)
            D_H_fake = disc_H(fake_pan.detach())
            D_H_real_loss = MSE(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = MSE(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_ms = gen_Z(noise_ms)
            D_Z_real = disc_Z(ms)
            D_Z_fake = disc_Z(fake_ms.detach())
            D_Z_real_loss = MSE(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练生成器 H Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_pan)
            D_Z_fake = disc_Z(fake_ms)
            loss_G_H = MSE(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = MSE(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_ms = gen_Z(fake_pan)
            cycle_pan = gen_H(fake_ms)
            cycle_ms_loss = L1(ms, cycle_ms)
            cycle_pan_loss = L1(pan, cycle_pan)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # 防止生成一个一样的？？
            identity_ms = gen_Z(ms)
            identity_pan = gen_H(pan)
            identity_ms_loss = L1(ms, identity_ms)
            identity_pan_loss = L1(pan, identity_pan)

            G_loss = (   # 6个loss
                loss_G_H
                + loss_G_Z
                + cycle_ms_loss * config1.LAMBDA_CYCLE
                + cycle_pan_loss * config1.LAMBDA_CYCLE
                + identity_ms_loss * config1.LAMBDA_IDENTITY
                + identity_pan_loss * config1.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        if os.path.exists(config.data_city+"_gan_saved_images0") == 0:
            os.makedirs(config.data_city+"_gan_saved_images0")
        if idx % 1 == 0:
            save_image(fake_pan, config.data_city+f"_gan_saved_images0/fake_pan_{idx}.png")
            save_image(fake_ms, config.data_city+f"_gan_saved_images0/fake_ms_{idx}.png")
            save_image(pan, config.data_city+f"_gan_saved_images0/pan_{idx}.png")
            save_image(ms, config.data_city+f"_gan_saved_images0/ms_{idx}.png")

        fake_pan = fake_pan.cpu().detach().numpy()
        fake_ms = fake_ms.cpu().detach().numpy()
        if idx == 0:
            test_ms = fake_pan
            test_pan = fake_ms
        else:
            test_ms = np.append(test_ms, fake_pan, axis=0)
            test_pan = np.append(test_pan, fake_ms, axis=0)

        loop.set_postfix(H_real=D_H_real_loss/(idx+1), H_fake=D_H_fake_loss/(idx+1), epoch=epoch)
    return test_ms, test_pan