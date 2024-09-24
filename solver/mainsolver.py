from solver.basesolver import BaseSolver
from utils.utils import make_optimizer, make_loss, make_scheduler, save_checkpoint, load_model
import os, torch, time, cv2, importlib
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from indicators.kappa import aa_oa, expo_result


class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.model = None
        self.cur_model = None
        self.train_time = 0
        self.test_time = 0
        self.matrix = None

        if self.cfg['train']['pretrained']:
            net_name = self.cfg['model_name'].lower()
            lib = importlib.import_module('model.' + net_name)
            net = lib.Net
            self.model = net(args=self.cfg)
            self.optimizer = make_optimizer(self.cfg, self.model.parameters())
            self.loss = make_loss(self.cfg['schedule']['loss'], self.cfg)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
            self.scheduler = make_scheduler(self.optimizer, self.cfg)

    def init_model(self):
        net_name = self.cfg['model_name'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        self.model = net(args=self.cfg)
        self.optimizer = make_optimizer(self.cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'], self.cfg)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.scheduler = make_scheduler(self.optimizer, self.cfg)

    def train(self):
        time1 = time.time()
        best_loss = float('inf') if self.cfg['train']['save_best'] else None
        best_epoch = 0 if self.cfg['train']['save_best'] else None
        self.init_model() if not self.cfg['train']['pretrained'] else None
        self.cur_model = self.model.to(self.DEVICE)
        while self.epoch < self.EPOCH:
            self.cur_model.train()
            train_loader = self.train_loader if self.cfg['nohup'] else tqdm(self.train_loader, leave=True)
            for batch_idx, (data1, data2, target, _, _) in enumerate(train_loader):
                data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                self.optimizer.zero_grad()
                output = self.cur_model(data1, data2)
                loss = self.loss(output, target.long())
                loss.backward()
                self.optimizer.step()
                print("{} times {}th epoch is trained".format(self.time, self.epoch)) \
                    if self.cfg['nohup'] else \
                    train_loader.set_postfix(ls=loss.item(), b_ep=best_epoch, ep=self.epoch,
                                             tm=self.time, m='train', d=self.cfg['device'])
            self.scheduler.step() if self.cfg['schedule']['if_scheduler'] else None
            train_loader.close() if not self.cfg['nohup'] else None
            if self.cfg['train']['save_best']:
                self.cur_model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    valid_loader = self.valid_loader if self.cfg['nohup'] else tqdm(self.valid_loader, leave=True)
                    for batch_idx, (data1, data2, target, _, _) in enumerate(valid_loader):
                        data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                        output = self.cur_model(data1, data2)
                        loss = self.loss(output, target.long())
                        val_loss += loss.item() * data1.size(0)
                        valid_loader.set_postfix(b_ls=best_loss, ls=val_loss, ep=self.epoch, tm=self.time,
                                                 m='valid', d=self.cfg['device']) if not self.cfg['nohup'] else None
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
            for batch_idx, (data1, data2, target, _, _) in enumerate(test_loader):
                # upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
                # self.xianhua(upsample(data1), 'm.png')
                # self.xianhua(data2, 'p.png')
                data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                output = self.cur_model(data1, data2)
                from sklearn.manifold import TSNE
                import matplotlib.pyplot as plt
                
                output_np = output.cpu().detach().numpy()
                target_np = target.cpu().detach().numpy()
                
                # 创建 t-SNE 模型并将数据降维到 2 维
                
                num_classes = 11
                # 创建颜色映射
                tsne = TSNE(n_components=2, random_state=42)
                output_tsne = tsne.fit_transform(output_np)
                
                # 创建颜色映射
                colors = plt.cm.get_cmap('tab10', num_classes)  # 使用Matplotlib的tab10颜色映射
                
                # 可视化降维后的数据，每个类别用不同颜色
                plt.figure(figsize=(8, 6))
                for i in range(num_classes):
                    plt.scatter(output_tsne[target_np == i, 0], output_tsne[target_np == i, 1],
                                color=colors(i), label=f'Class {i}', alpha=0.6)
                
                # plt.title('t-SNE Visualization of Neural Network Output')
                # plt.xlabel('Component 1')
                # plt.ylabel('Component 2')
                # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # 将图例放在右上角外部
                plt.savefig("{}pan.jpg".format(self.time))

                # test_loss += nn.CrossEntropyLoss(output, target.long())
                pred = output.data.max(1, keepdim=True)[1]
                for i in range(len(target)):
                    test_matrix[int(pred[i].item())][int(target[i].item())] += 1
                break
            print("test down") if self.cfg['nohup'] else test_loader.set_postfix(mode='test')
        test_loader.close() if not self.cfg['nohup'] else None
        time2 = time.time()
        self.test_time = time2 - time1
        self.test_matrix = test_matrix
        self.indicator()
        
    # def indicator(self):
    #     savepath = self.cfg['RESULT_output'] + str(self.time) + "_matrix.npy" if self.cfg['test']['save_matrix'] else None
    #     np.save(savepath, self.test_matrix)
    #     result = aa_oa(self.test_matrix)
    #     expo_result(result, self.cfg['RESULT_excel'], [self.train_time, self.test_time], self.time)
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
                for batch_idx, (data1, data2, target, x, y) in enumerate(color_loader1):
                    data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                    out = self.cur_model(data1, data2)
                    pred = out.data.max(1, keepdim=True)[1]
                    for i in range(int(data1.shape[0])):
                        label_np1[int(x[i])][int(y[i])] = int(pred[i])
                        label_np2[int(x[i])][int(y[i])] = int(pred[i])
                    color_loader1.set_postfix(mode='verify') if not self.cfg['nohup'] else None
                color_loader1.close() if not self.cfg['nohup'] else None
            if self.cfg['color']['unsupervised']:
                color_loader2 = self.color_loader2 if self.cfg['nohup'] else tqdm(self.color_loader2, leave=True)
                for batch_idx, (data1, data2, target, x, y) in enumerate(color_loader2):
                    data1, data2, target = data1.to(self.DEVICE), data2.to(self.DEVICE), target.to(self.DEVICE)
                    out = self.cur_model(data1, data2)
                    pred = out.data.max(1, keepdim=True)[1]
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
        savepath = self.cfg['RESULT_output'] + str(self.time) + "_pic_1.png"
        picture1.save(savepath) if self.cfg['color']['supervised'] else None
        
        for i in range(label_np2.shape[0]):
            for j in range(label_np2.shape[1]):
                label_pic[i][j] = self.cfg['DATA_DICT'][self.cfg['data_city']]['color'][int(label_np2[i][j])]
        picture2 = Image.fromarray(np.uint8(label_pic))
        # picture.show()
        savepath = self.cfg['RESULT_output'] + str(self.time) + "_pic_2.png"
        picture2.save(savepath) if self.cfg['color']['supervised'] else None
            
    def run(self):
        while self.time < self.TIME:
            self.dataloader()
            self.train() if self.cfg['train']['index'] else None
            self.test() if self.cfg['test']['index'] else None
            self.color() if self.cfg['color']['index'] else None
            self.time += 1

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
            
    def proof(self):
        ps = 64
        x = 200
        y = 200
        m = self.MS[x:x+ps, y:y+ps, :]
        from function.function import to_tensor
        m = to_tensor(m)
        vmin, vmax = m.min(), m.max()
        p = self.PAN[x*4:(x+ps)*4, y*4:(y+ps)*4]
        p = to_tensor(p)
        
        m1 = m[:, :, 0]
        m2 = m[:, :, 1]
        m3 = m[:, :, 2]
        m4 = m[:, :, 3]

        
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        from skimage.measure import block_reduce
        from scipy.stats import norm
        # ## 4谱段图像演示
        # plt.figure(figsize=(10, 8))
        # for i in range(4):
        #     plt.subplot(2, 2, i+1)
        #     plt.title("channel {} Heatmap".format(i))
        #     sns.heatmap(m[:, :, i], cmap="viridis", annot=False, vmin=vmin, vmax=vmax)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
        #     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        #     plt.xticks([])  # 隐藏x轴刻度标签
        #     plt.yticks([])
        # # plt.xlabel("Columns")
        # # plt.ylabel("Rows")
        # plt.show()

        # ## 4谱段特征分析
        # plt.figure(figsize=(8, 6))
        # plt.title("")
        # plt.hist(m1.flatten(), color='blue', bins=90, alpha=0.9, label='channel 0')
        # plt.hist(m2.flatten(), color='green', bins=90, alpha=0.9, label='channel 1')
        # plt.hist(m3.flatten(), color='red', bins=90, alpha=0.9, label='channel 2')
        # plt.hist(m4.flatten(), color='lightcoral', bins=90, alpha=0.9, label='channel 3')
        # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        # plt.xticks([])  # 隐藏x轴刻度标签
        # plt.yticks([])
        # plt.legend()
        # plt.show()

        ## pan与I分量对比
        # plt.figure(figsize=(8, 6))
        # plt.title("")
        # plt.hist((np.mean(m, axis=2)).flatten(), bins=90, alpha=0.9, label='mean')
        # # plt.hist((np.mean(m[:, :, :3], axis=2)).flatten(), bins=90, alpha=0.9, label='rgb mean')
        # plt.hist(block_reduce(p, (4, 4), func=np.mean).flatten(), bins=90, alpha=0.9, label='pan')
        # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        # plt.xticks([])  # 隐藏x轴刻度标签
        # plt.yticks([])
        # plt.legend()
        # plt.show()

        from model.contourlet_torch import ContourDec
        _, p_1 = ContourDec(3)(torch.from_numpy(p[np.newaxis, np.newaxis, :, :]/1.0).type(torch.FloatTensor))
        _, m_1 = ContourDec(3)(torch.from_numpy(np.mean(m, axis=2)[np.newaxis, np.newaxis, :, :]/1.0).type(torch.FloatTensor))
        # print(p1.shape, m1.shape)

        # plt.figure(figsize=(10, 8))
        # for i in range(8):
        #     plt.subplot(2, 4, i + 1)
        #     plt.title("ms direction {} Heatmap".format(i))
        #     sns.heatmap(torch.relu(torch.log(m_1[0, i, :, :])), cmap="Greys", annot=False)
        #                 #vmin=torch.relu(torch.log(m_1.min())), vmax=torch.relu(torch.log(m_1.max())))  # cmap可以选择其他颜色映射，annot为True会显示数值标签
        #     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        #     plt.xticks([])  # 隐藏x轴刻度标签
        #     plt.yticks([])
        # # plt.xlabel("Columns")
        # # plt.ylabel("Rows")
        # plt.show()

        # plt.figure(figsize=(10, 8))
        # for i in range(8):
        #     plt.subplot(2, 4, i + 1)
        #     plt.title("pan direction {} Heatmap".format(i))
        #     sns.heatmap(torch.relu(torch.log(p_1[0, i, :, :])), cmap="Greys", annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
        #     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
        #     plt.xticks([])  # 隐藏x轴刻度标签
        #     plt.yticks([])
        # # plt.xlabel("Columns")
        # # plt.ylabel("Rows")
        # plt.show()

        p_l, p_1 = ContourDec(2)(torch.from_numpy(p[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))
        p_l1, p_2 = ContourDec(2)(p_l)
        _, p_3 = ContourDec(2)(p_l1)
        # _, m_1 = ContourDec(2)(torch.from_numpy(np.mean(m, axis=2)[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))

        plt.figure(figsize=(10, 8))
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 4, i*4 + j + 1)
                plt.title("pan d {} ".format(i*4 + j + 1))
                sns.heatmap(torch.sqrt(p_1[0, i, :, :] * p_1[0, j, :, :]), cmap="viridis", annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
                plt.xticks([])  # 隐藏x轴刻度标签
                plt.yticks([])
                # plt.xlabel("Columns")
                # plt.ylabel("Rows")
        plt.show()

        plt.figure(figsize=(10, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.title("pan d {} ".format(i))
            sns.heatmap(p_1[0, i, :, :], cmap="viridis",annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
            plt.xticks([])  # 隐藏x轴刻度标签
            plt.yticks([])
                # plt.xlabel("Columns")
                # plt.ylabel("Rows")
        plt.show()

        plt.figure(figsize=(10, 8))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.title("pan d {} ".format(i))
            sns.heatmap(p_2[0, i, :, :], cmap="viridis", annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
            plt.xticks([])  # 隐藏x轴刻度标签
            plt.yticks([])
            # plt.xlabel("Columns")
            # plt.ylabel("Rows")
        plt.show()

        plt.figure(figsize=(10, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title("pan d {} ".format(i))
            sns.heatmap(p_3[0, i, :, :], cmap="viridis", annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
            plt.xticks([])  # 隐藏x轴刻度标签
            plt.yticks([])
            # plt.xlabel("Columns")
            # plt.ylabel("Rows")
        plt.show()

        _, m_1 = ContourDec(2)(torch.from_numpy(m1[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))
        _, m_2 = ContourDec(2)(torch.from_numpy(m2[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))
        _, m_3 = ContourDec(2)(torch.from_numpy(m3[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))
        _, m_4 = ContourDec(2)(torch.from_numpy(m4[np.newaxis, np.newaxis, :, :] / 1.0).type(torch.FloatTensor))

        ms_1 = torch.cat((m_1, m_2, m_3, m_4), dim=1)
        plt.figure(figsize=(10, 8))
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 4, i * 4 + j + 1)
                plt.title("ms d {} ".format(i * 4 + j + 1))
                sns.heatmap(ms_1[0, i * 4 + j, :, :], cmap="viridis", annot=False)  # cmap可以选择其他颜色映射，annot为True会显示数值标签
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # 隐藏刻度线
                plt.xticks([])  # 隐藏x轴刻度标签
                plt.yticks([])
                # plt.xlabel("Columns")
                # plt.ylabel("Rows")
        plt.show()

    def visualize_channels(self, tensor, num_channels=8, cols=4):
        """
        可视化指定数量的通道。
        :param tensor: BCHW 形状的张量。
        :param num_channels: 要展示的通道数量。
        :param cols: 每行显示的图像数量。
        """
        tensor = tensor[0]  # 选择批次中的第一个样本
        rows = num_channels // cols + int(num_channels % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten()

        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(tensor[i].cpu().detach().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Channel {i + 1}')

        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


def equalize_histogram(band):
    hist, bins = np.histogram(band.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[band]