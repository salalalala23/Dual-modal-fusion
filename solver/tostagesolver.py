from solver.mainsolver import Solver
from utils.utils import make_optimizer, make_loss, make_scheduler
import torch.nn.functional as F
import os, torch, time, cv2, importlib
from tqdm import tqdm
import numpy as np
from PIL import Image
from indicators.kappa import aa_oa, expo_result
from function.function import data_process_dqtl_stage1, pan_pic_product, data_padding, data_show, split_data_old, to_tensor
from model.discriminator import Discriminator2
from model.generator import Generator
import torch.optim as optim
from utils.utils import load_checkpoint, save_checkpoint, save_point_sche, load_model
from train.dataset import FusionDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from train.dataset import dataset_qua_dqtl


class toStageSolver(Solver):
    def __init__(self, cfg):
        super().__init__(cfg)
        # super(Solver, self).__init__(cfg)
        self.disc_P = None
        self.disc_M = None
        self.gen_M = None
        self.gen_P = None
        self.opt_disc = None
        self.opt_gen = None
        self.MSE = make_loss(self.cfg['dqtl']['loss1'], cfg)
        self.L1 = make_loss(self.cfg['dqtl']['loss2'], cfg)
        self.onestage_tl = None
        self.onestage_vl = None
        self.g_scaler = None
        self.d_scaler = None
        self.test_ms = None
        self.test_pan = None
        self.ms_gan = None
        self.pan_gan = None

        self.init_stage1_model()
        if self.cfg['dqtl']['load_model']:
            # print(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gH'])
            load_model(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gH'],
                            self.gen_P, self.DEVICE)
            load_model(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gZ'],
                            self.gen_M, self.DEVICE)
            load_model(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dH'],
                            self.disc_P, self.DEVICE)
            load_model(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dZ'],
                            self.disc_M, self.DEVICE)

        if self.cfg['train']['pretrained']:
            net_name = self.cfg['model_name'].lower()
            lib = importlib.import_module('model.' + net_name)
            net = lib.Net
            self.model = net(args=self.cfg)
            self.optimizer = make_optimizer(self.cfg, self.model.parameters())
            self.loss = make_loss(self.cfg['schedule']['loss'], self.cfg)
            self.scheduler = make_scheduler(self.optimizer, self.cfg)

    def init_stage1_model(self):
        net_name = self.cfg['dqtl']['encoder_name'].lower()
        lib = importlib.import_module('model.' + net_name)
        self.disc_P = lib.Net().to(self.DEVICE)
        self.disc_M = lib.Net().to(self.DEVICE)
        
        net_name = self.cfg['dqtl']['decoder_name'].lower()
        lib = importlib.import_module('model.' + net_name)
        self.gen_M = lib.Net().to(self.DEVICE)
        self.gen_P = lib.Net().to(self.DEVICE)
        
        self.opt_disc = optim.Adam(
            list(self.disc_P.parameters()) + list(self.disc_M.parameters()),
            lr=self.cfg['dqtl']['lr'],
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            list(self.gen_P.parameters()) + list(self.gen_M.parameters()),
            lr=self.cfg['dqtl']['lr'],
            betas=(0.5, 0.999),
        )
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        
    def gan(self, data):
        import matplotlib.pyplot as plt
        Ld_m, Ld_p, Lg_m, Lg_p, Lc_m, Lc_p, Li_m, Li_p, La_m, La_p = [], [], [], [], [], [], [], [], [], []
        for epoch in range(self.cfg['dqtl']['epochs']):
            train_loader = data if self.cfg['nohup'] else tqdm(data, leave=True)
            ld_m, ld_p, lg_m, lg_p, lc_m, lc_p, li_m, li_p, la_m, la_p = [], [], [], [], [], [], [], [], [], []
            for idx, (m, p, _) in enumerate(train_loader):
                p = p.to(self.DEVICE)
                m = m.to(self.DEVICE)
                n_m = torch.randn_like(m).to(self.DEVICE)
                n_p = torch.randn_like(p).to(self.DEVICE)

                with torch.cuda.amp.autocast():
                    fake_pan = self.gen_P(n_m)
                    D_P_real = self.disc_P(p)
                    D_P_fake = self.disc_P(fake_pan.detach())
                    D_P_real_loss = self.MSE(D_P_real, torch.ones_like(D_P_real))
                    D_P_fake_loss = self.MSE(D_P_fake, torch.zeros_like(D_P_fake))
                    D_P_loss = D_P_real_loss + D_P_fake_loss

                    fake_ms = self.gen_M(n_p)
                    D_M_real = self.disc_M(m)
                    D_M_fake = self.disc_M(fake_ms.detach())
                    D_M_real_loss = self.MSE(D_M_real, torch.ones_like(D_M_real))
                    D_M_fake_loss = self.MSE(D_M_fake, torch.zeros_like(D_M_fake))
                    D_M_loss = D_M_real_loss + D_M_fake_loss
                    D_loss = (D_P_loss + D_M_loss)
                self.opt_disc.zero_grad()
                self.d_scaler.scale(D_loss).backward()
                self.d_scaler.step(self.opt_disc)
                self.d_scaler.update()

                # 训练生成器 H Z
                with torch.cuda.amp.autocast():
                    # adversarial loss for both generators
                    D_P_fake = self.disc_P(fake_pan)
                    D_M_fake = self.disc_M(fake_ms)
                    loss_G_P = self.MSE(D_P_fake, torch.ones_like(D_P_fake))
                    loss_G_M = self.MSE(D_M_fake, torch.ones_like(D_M_fake))

                    # cycle loss
                    cycle_ms = self.gen_M(fake_pan)
                    cycle_pan = self.gen_P(fake_ms)
                    cycle_ms_loss = self.L1(n_m, cycle_ms)
                    cycle_pan_loss = self.L1(n_p, cycle_pan)

                    # identity loss (remove these for efficiency if you set lambda_identity=0)
                    
                    identity_ms = self.gen_M(n_m)
                    identity_pan = self.gen_P(n_p)
                    identity_ms_loss = self.L1(n_m, identity_ms)
                    identity_pan_loss = self.L1(n_p, identity_pan)

                    # adversarial loss
                    adversarial_pan = self.gen_P(cycle_ms)
                    adversarial_ms = self.gen_M(cycle_pan)
                    adversarial_pan_loss = self.L1(n_m, adversarial_ms)
                    adversarial_ms_loss = self.L1(n_p, adversarial_pan)
                    G_loss = (  # 6个loss
                            loss_G_P
                            + loss_G_M
                            # + (1 - torch.mean(torch.cosine_similarity(p/m, p/fake_ms, eps=1e-8)))
                            # + (1 - torch.mean(torch.cosine_similarity(p/m, fake_pan/m, eps=1e-8)))
                            + cycle_ms_loss * self.cfg['dqtl']['l_cy']
                            + cycle_pan_loss * self.cfg['dqtl']['l_cy']
                            + identity_ms_loss * self.cfg['dqtl']['l_id']
                            + identity_pan_loss * self.cfg['dqtl']['l_id']
                            + adversarial_ms_loss * self.cfg['dqtl']['l_ad']
                            + adversarial_pan_loss * self.cfg['dqtl']['l_ad']
                    )/2
                self.opt_gen.zero_grad()
                self.g_scaler.scale(G_loss).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()
                if os.path.exists(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + "saved_images") == 0:
                    os.makedirs(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + "saved_images")
                if idx % 1 == 0 and (epoch == 0 or epoch % 20 == 19):
                    self.xianhua(fake_pan, self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + f"saved_images/f_p_{epoch}_{idx}.png")
                    self.xianhua(fake_ms, self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + f"saved_images/f_m_{epoch}_{idx}.png")
                    self.xianhua(p, self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + f"saved_images/p_{idx}.png")
                    self.xianhua(m, self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS'] + f"saved_images/m_{idx}.png")

                fake_pan = fake_pan.cpu().detach().numpy()
                fake_ms = fake_ms.cpu().detach().numpy()

                self.test_ms = fake_pan if idx == 0 else np.append(self.test_ms, fake_pan, axis=0)
                self.test_pan = fake_ms if idx == 0 else np.append(self.test_pan, fake_ms, axis=0)

                if self.cfg['nohup']:
                    print("stage1 {} {} epoch is trained".format(epoch, idx))
                else:
                    train_loader.set_postfix(H_real=(D_P_real_loss / (idx + 1)).item(),
                                             H_fake=(D_P_fake_loss / (idx + 1)).item(), epoch=epoch)


    def train_stage1(self):
        # shape is (2001, 2101, 4)
        data, index_x, index_y = data_process_dqtl_stage1(self.MS, self.PAN, self.cfg)
        # self.init_stage1_model()
        # if self.cfg['dqtl']['load_model']:
        #     # print(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gH'])
        #     load_checkpoint(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gH'],
        #                     self.gen_P, self.opt_gen, self.cfg['dqtl']['lr'], self.DEVICE)
        #     load_checkpoint(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gZ'],
        #                     self.gen_M, self.opt_gen, self.cfg['dqtl']['lr'], self.DEVICE)
        #     load_checkpoint(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dH'],
        #                     self.disc_P, self.opt_disc, self.cfg['dqtl']['lr'], self.DEVICE)
        #     load_checkpoint(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dZ'],
        #                     self.disc_M, self.opt_disc, self.cfg['dqtl']['lr'], self.DEVICE)

        train_dataset = FusionDataset(data)
        # val_dataset = FusionDataset(data)
        self.onestage_tl = DataLoader(
            train_dataset,
            batch_size=self.cfg['dqtl']['batch_size'],
            shuffle=False,
            num_workers=self.cfg['dqtl']['num_workers'],
            pin_memory=True,
        )
        # self.onestage_vl = DataLoader(
        #     val_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     pin_memory=True,
        # )

        self.gan(self.onestage_tl)

        # if os.path.exists(self.cfg['dqtl']['WEIGHTS']) == 0:
        #     os.makedirs(self.cfg['dqtl']['WEIGHTS'])
        if self.cfg['dqtl']['save_model']:
            save_checkpoint(self.gen_P, self.opt_gen,
                            filename=self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gH'])
            save_checkpoint(self.gen_M, self.opt_gen,
                            filename=self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_gZ'])
            save_checkpoint(self.disc_P, self.opt_disc,
                            filename=self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dH'])
            save_checkpoint(self.disc_M, self.opt_disc,
                            filename=self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+self.cfg['dqtl']['check_dZ'])
        self.ms_gan = pan_pic_product(self.test_ms, [index_x, index_y], self.cfg)
        self.pan_gan = pan_pic_product(self.test_pan, [index_x, index_y], self.cfg)
        print(self.ms_gan.shape, self.pan_gan.shape)
        self.xianhua(torch.from_numpy(self.ms_gan).type(torch.FloatTensor).unsqueeze(0),
                     self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+f"fake_ms.png")
        self.xianhua(torch.from_numpy(self.pan_gan).type(torch.FloatTensor).unsqueeze(0),
                   self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+f"fake_pan.png")
        # self.xianhua(torch.from_numpy(self.ms_gan).type(torch.FloatTensor),
        #            self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+f"fake_ms.png")
        # self.xianhua(torch.from_numpy(self.pan_gan).type(torch.FloatTensor),
        #            self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+f"fake_pan.png")
        self.ms_gan = self.ms_gan.transpose((1, 2, 0))
        self.pan_gan = self.pan_gan.transpose((1, 2, 0))
        np.save(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+'msgan.npy', self.ms_gan)
        np.save(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+'pangan.npy', self.pan_gan)
    
    def train_stage2(self):
        if self.cfg['dqtl']['pre_trained']:
            self.ms_gan = np.load(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+'msgan.npy')
            self.pan_gan = np.load(self.cfg['expo_result'] + self.cfg['dqtl']['WEIGHTS']+'pangan.npy')
        else:
            self.train_stage1()
        MS = self.MS
        PAN = np.load(self.cfg['data_address'] + '/pan.npy')
        # print(MS.shape, PAN.shape, self.ms_gan.shape, self.pan_gan.shape)
        ms, pan, ms_gan, pan_gan = data_padding(MS, self.cfg, 'ms'),\
            data_padding(PAN, self.cfg, 'ms'), \
            data_padding(self.ms_gan, self.cfg, 'ms'), \
            data_padding(self.pan_gan, self.cfg, 'ms')
        label_np = np.load(self.cfg['data_address'] + 'label.npy', encoding='bytes', allow_pickle=True)
        data_show(label_np)
        xyl_matrix, self.matrix_ = split_data_old(label_np, self.cfg)
        self.dataset = dataset_qua_dqtl(ms, pan, ms_gan, pan_gan, xyl_matrix, self.cfg)

    def train(self):
        time1 = time.time()
        best_loss = float('inf') if self.cfg['train']['save_best'] else None
        best_epoch = 0 if self.cfg['train']['save_best'] else None
        self.init_model() if not self.cfg['train']['pretrained'] else None
        self.cur_model = self.model.to(self.DEVICE)
        while self.epoch < self.EPOCH:
            self.cur_model.train()
            train_loader = self.train_loader if self.cfg['nohup'] else tqdm(self.train_loader, leave=True)
            for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(train_loader):
                data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
                data = torch.concat([data1, data2, data3, data4])
                bs = len(data1)
                self.optimizer.zero_grad()
                output = self.cur_model(data)# self.gen_P, self.gen_M)
                loss = self.loss(output, bs, target, self.cfg)
                # loss = self.loss(output, target.long())
                loss.backward()
                self.optimizer.step()
                print("{} times {}th epoch is trained".format(self.time, self.epoch)) \
                    if self.cfg['nohup'] else \
                    train_loader.set_postfix(loss=loss.item(), best_epoch=best_epoch,
                                             epoch=self.epoch, time=self.time, mode='train')
            self.scheduler.step() if self.cfg['schedule']['if_scheduler'] else None
            train_loader.close() if not self.cfg['nohup'] else None
            if self.cfg['train']['save_best']:
                self.cur_model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    valid_loader = self.valid_loader if self.cfg['nohup'] else tqdm(self.valid_loader, leave=True)
                    for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(valid_loader):
                        data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                        data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
                        data = torch.concat([data1, data2, data3, data4])
                        bs = len(data1)
                        output = self.cur_model(data)# self.gen_P, self.gen_M)
                        loss = self.loss(output, bs, target, self.cfg)
                        # loss = self.loss(output, target.long())
                        val_loss += loss.item() * data1.size(0)
                        valid_loader.set_postfix(best_loss=best_loss, loss=val_loss, epoch=self.epoch,
                                                 time=self.time, mode='valid') if not self.cfg['nohup'] else None
                        if val_loss > best_loss:
                            break
                    valid_loader.close() if self.cfg['nohup'] else None
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = self.epoch
                    best_weights = self.cur_model.state_dict()
                    torch.save(best_weights, self.cfg['RESULT_output'] + str(self.time) + '_weights.pth')
                    print("best epoch now is {}".format(self.epoch)) if self.cfg['nohup'] else None
            save_checkpoint(self.cur_model, self.optimizer, 
                            self.cfg['RESULT_output'] + str(self.time) + '_curweights.pth')
            self.epoch += 1
        time2 = time.time()
        self.train_time = time2 - time1
        self.epoch = 0

    def test(self):
        time1 = time.time()
        if not self.cfg['train']['index']:
            self.init_model()
            self.cur_model = self.model.to(self.DEVICE)
        if self.cfg['train']['save_best']:
            self.cur_model.load_state_dict(torch.load(self.cfg['RESULT_output'] + str(self.time) + '_weights.pth'))
        else:
            self.cur_model.load_state_dict(torch.load(self.cfg['RESULT_output'] + str(self.time) + '_curweights.pth'))
        self.cur_model.eval()
        test_loss = 0
        test_matrix = np.zeros([self.cfg['Categories_Number'], self.cfg['Categories_Number']])
        with torch.no_grad():
            test_loader = self.test_loader if self.cfg['nohup'] else tqdm(self.test_loader, leave=True)
            for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(test_loader):
                data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
                bs = len(data1)
                data = torch.concat([data1, data2, data3, data4])
                output = self.cur_model(data)# self.gen_P, self.gen_M)
                pred = (output[:bs]+output[bs:2*bs]).softmax(dim=-1).data.max(1, keepdim=True)[1]
                # pred = output[:bs].data.max(1, keepdim=True)[1]
                # test_loss += nn.CrossEntropyLoss(output, target.long())
                for i in range(len(target)):
                    test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            print("test down") if self.cfg['nohup'] else test_loader.set_postfix(mode='test')
        test_loader.close() if not self.cfg['nohup'] else None
        time2 = time.time()
        self.test_time = time2 - time1
        self.test_matrix = test_matrix
        self.indicator()

    def color(self):
        if not self.cfg['train']['index'] and not self.cfg['test']['index']:
            self.init_model()
            self.cur_model = self.model.to(self.DEVICE)
        self.cur_model.load_state_dict(torch.load(self.cfg['RESULT_output'] + str(self.time) + '_weights.pth'))
        self.cur_model.eval()
        size = self.cfg['DATA_DICT'][self.cfg['data_city']]['size']
        label_np1 = np.zeros([size[0], size[1]])
        label_np2 = np.zeros([size[0], size[1]])
        with torch.no_grad():
            if self.cfg['color']['supervised']:
                color_loader1 = self.color_loader1 if self.cfg['nohup'] else tqdm(self.color_loader1, leave=True)
                for batch_idx, (data1, data2, data3, data4, target, x, y) in enumerate(color_loader1):
                    data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                    data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
                    bs = len(data1)
                    data = torch.concat([data1, data2, data3, data4])
                    output = self.cur_model(data)# self.gen_P, self.gen_M)
                    pred = (output[:bs]+output[bs:2*bs]).softmax(dim=-1).data.max(1, keepdim=True)[1]
                    for i in range(int(data1.shape[0])):
                        label_np1[int(x[i])][int(y[i])] = int(pred[i])
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    color_loader1.set_postfix(mode='verify') if not self.cfg['nohup'] else None
                color_loader1.close() if not self.cfg['nohup'] else None
            if self.cfg['color']['unsupervised']:
                color_loader2 = self.color_loader2 if self.cfg['nohup'] else tqdm(self.color_loader2, leave=True)
                for batch_idx, (data1, data2, data3, data4, target, x, y) in enumerate(color_loader2):
                    data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                    data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
                    bs = len(data1)
                    data = torch.concat([data1, data2, data3, data4])
                    output = self.cur_model(data)#self.gen_P, self.gen_M)
                    pred = (output[:bs]+output[bs:2*bs]).softmax(dim=-1).data.max(1, keepdim=True)[1]
                    for i in range(int(data1.shape[0])):
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    color_loader2.set_postfix(mode='verify') if not self.cfg['nohup'] else None
                color_loader2.close() if not self.cfg['nohup'] else None
        label_pic = np.zeros([size[0], size[1], 3])
        for i in range(label_np1.shape[0]):
            for j in range(label_np1.shape[1]):
                label_pic[i][j] = self.cfg['DATA_DICT'][self.cfg['data_city']]['color'][int(label_np1[i][j])]
        picture1 = Image.fromarray(np.uint8(label_pic))
        # picture.show()
        savepath = self.cfg['RESULT_output'] + str(self.time) + "_pic_1.jpg"
        picture1.save(savepath) if self.cfg['color']['supervised'] else None

        for i in range(label_np2.shape[0]):
            for j in range(label_np2.shape[1]):
                label_pic[i][j] = self.cfg['DATA_DICT'][self.cfg['data_city']]['color'][int(label_np2[i][j])]
        picture2 = Image.fromarray(np.uint8(label_pic))
        # picture.show()
        savepath = self.cfg['RESULT_output'] + str(self.time) + "_pic_2.jpg"
        picture2.save(savepath) if self.cfg['color']['supervised'] else None

    def dual_model_generation(self):
        self.dataloader()
        self.gan(self.train_loader)

    def run(self):
        self.train_stage2()
        while self.time < self.TIME:
            self.dataloader()
            self.train() if self.cfg['train']['index'] else None
            self.test() if self.cfg['test']['index'] else None
            self.color() if self.cfg['color']['index'] else None
            self.time += 1

    def visualize_extract(self):
        self.train_stage2()
        self.dataloader()
        self.init_model() if not self.cfg['train']['pretrained'] else None
        self.cur_model = self.model.to(self.DEVICE)
        train_loader = self.train_loader if self.cfg['nohup'] else tqdm(self.train_loader, leave=True)
        for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(train_loader):
            data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
            data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
            data = torch.concat([data1, data2, data3, data4])
            bs = len(data1)
            self.optimizer.zero_grad()
            output = self.cur_model(data)  # self.gen_P, self.gen_M)
            save_image(output[:bs], self.cfg['RESULT_output'] + f"train_ms{batch_idx}.png")
            save_image(output[bs:2 * bs], self.cfg['RESULT_output'] + f"train_pan{batch_idx}.png")
            save_image(output[2 * bs:3 * bs], self.cfg['RESULT_output'] + f"train_gm{batch_idx}.png")
            save_image(output[-bs:], self.cfg['RESULT_output'] + f"train_gp{batch_idx}.png")
        if self.cfg['train']['save_best']:
            self.cur_model.load_state_dict(torch.load(self.cfg['RESULT_output'] + str(self.time) + '_weights.pth'))
        else:
            self.cur_model.load_state_dict(torch.load(self.cfg['RESULT_output'] + str(self.time) + '_curweights.pth'))
        for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(train_loader):
            data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
            data3, data4 = data3.to(self.DEVICE), data4.to(self.DEVICE)
            data = torch.concat([data1, data2, data3, data4])
            bs = len(data1)
            self.optimizer.zero_grad()
            output = self.cur_model(data)  # self.gen_P, self.gen_M)
            self.xianhua(output[:bs], self.cfg['RESULT_output'] + f"test_ms{batch_idx}.png")
            self.xianhua(output[bs:2 * bs], self.cfg['RESULT_output'] + f"test_pan{batch_idx}.png")
            self.xianhua(output[2 * bs:3 * bs], self.cfg['RESULT_output'] + f"test_gm{batch_idx}.png")
            self.xianhua(output[-bs:], self.cfg['RESULT_output'] + f"test_gp{batch_idx}.png")

    def visualize_deal(self):
        m =   np.mean(cv2.imread(self.cfg['RESULT_output'] + f"train_ms11.png"), axis=2)
        p =  np.mean(cv2.imread(self.cfg['RESULT_output'] + f"train_pan11.png"), axis=2)
        gm =  np.mean(cv2.imread(self.cfg['RESULT_output'] + f"train_gm11.png"), axis=2)
        gp =  np.mean(cv2.imread(self.cfg['RESULT_output'] + f"train_gp11.png"), axis=2)
        t_m =  np.mean(cv2.imread(self.cfg['RESULT_output'] + f"test_ms11.png"), axis=2)
        t_p = np.mean(cv2.imread(self.cfg['RESULT_output'] + f"test_pan11.png"), axis=2)
        t_gm = np.mean(cv2.imread(self.cfg['RESULT_output'] + f"test_gm11.png"), axis=2)
        t_gp = np.mean(cv2.imread(self.cfg['RESULT_output'] + f"test_gp11.png"), axis=2)

        bs = m.shape[0]
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        vectors = np.concatenate([m[:bs, :], p[:bs, :], gm[:bs, :], gp[:bs, :]], axis=0)

        tsne = TSNE(n_components=2)
        embedded_vectors = tsne.fit_transform(vectors)

        # 可视化聚类结果
        plt.scatter(embedded_vectors[:bs, 0], embedded_vectors[:bs, 1], label='M')
        plt.scatter(embedded_vectors[bs:2*bs, 0], embedded_vectors[bs:2*bs, 1], label='PAN')
        plt.scatter(embedded_vectors[2*bs:3*bs, 0], embedded_vectors[2*bs:3*bs, 1], label='GM')
        plt.scatter(embedded_vectors[3*bs:, 0], embedded_vectors[3*bs:, 1], label='GP')
        plt.legend()
        plt.axis('off')
        plt.show()

        vectors = np.concatenate([t_m[:bs, :], t_p[:bs, :], t_gm[:bs, :], t_gp[:bs, :]], axis=0)

        tsne = TSNE(n_components=2)
        embedded_vectors = tsne.fit_transform(vectors)

        # 可视化聚类结果
        plt.scatter(embedded_vectors[:bs, 0], embedded_vectors[:bs, 1], label='M')
        plt.scatter(embedded_vectors[bs:2 * bs, 0], embedded_vectors[bs:2 * bs, 1], label='PAN')
        plt.scatter(embedded_vectors[2 * bs:3 * bs, 0], embedded_vectors[2 * bs:3 * bs, 1], label='GM')
        plt.scatter(embedded_vectors[3 * bs:, 0], embedded_vectors[3 * bs:, 1], label='GP')
        plt.legend()
        plt.axis('off')
        plt.show()

    def xianhua(self, img, name="", equalize=1):
        img = img[0].cpu().detach().numpy()

        if img.shape[0] == 4:
            band_data = img[(2, 1, 0), :, :]
            scaled_data = []
            for i, band in enumerate(band_data):
                # band_min, band_max = self.MS[i].min(), self.MS[i].max()
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
            band_min, band_max = self.PAN.min(), self.PAN.max()

            processed_band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            if equalize:
                processed_band = equalize_histogram(processed_band)
            processed_array = processed_band

        else:
            raise ValueError("Unsupported image type. Please use 'multispectral' or 'pan'.")

        result = Image.fromarray(processed_array, 'RGB' if img.shape[0] == 4 else 'L')
        # result.show()
        result.save(name) if name != "" else None

def equalize_histogram(band):
    hist, bins = np.histogram(band.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[band]


