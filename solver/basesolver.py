import os, torch, time
from torch.utils.data import DataLoader, Subset
from train.dataset import *
from function.function import *
from indicators.kappa import aa_oa, expo_result


class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg['task']
        self.TIME = cfg['time']
        self.time = cfg['index']
        self.EPOCH = cfg['epoch']
        self.epoch = 0
        self.DEVICE = cfg['device']

        self.timestamp = int(time.time())

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.ms = read_tif(cfg, 'ms')
        self.pan = read_tif(cfg, 'pan')

        if cfg['data_new'] == 1:
            self.train_label = np.load(cfg['data_address'] + 'train.npy')
            self.test_label = np.load(cfg['data_address'] + 'test.npy')

        self.MS = data_padding(self.ms, cfg, 'ms')
        self.PAN = data_padding(self.pan, cfg, 'pan')

        if not os.path.exists(cfg['data_address'] + 'label.npy'):
            label_mat2np(cfg)
        label_np = np.load(cfg['data_address'] + 'label.npy', encoding='bytes', allow_pickle=True)
        data_show(label_np)
        if cfg['data_new'] == 1:
            xyl_matrix, self.traintest_index = split_data(self.train_label, self.test_label, label_np, cfg)
            _, self.matrix_ = split_data_old(label_np, cfg)
        else:
            xyl_matrix, self.matrix_ = split_data_old(label_np, cfg)

        if cfg['use_h5']:  # 未完成
            raise AttributeError("not finished")
            # if not os.path.exists(cfg['data_address'] + str(cfg['patch_size']) + "_train.h5"):
            #     dataset_cut(self.PAN, self.MS, xyl_matrix, self.matrix_[0], cfg, mode='train')
            #     dataset_cut(self.PAN, self.MS, xyl_matrix, self.matrix_[1], cfg, mode='color')
            # self.train_data = read_h5(cfg['data_address']+str(cfg['patch_size'])+"_train.h5")
            # print('Train dataset大小为:', self.train_data['ms'].shape)
            # self.dataset = dataset_h5(self.train_data)
            # print('All dataset大小为:', self.dataset.__len__())
            # if cfg['color']['index']:
            #     self.color_data = read_h5(cfg['data_address']+str(cfg['patch_size'])+"_train.h5")
            #     print('color dataset大小为:', self.color_data['ms'].shape)
        else:
            self.dataset = dataset_dual(self.MS, self.PAN, xyl_matrix, cfg)  # 总的数据集
            print('All dataset大小为:', self.dataset.__len__())

        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}

    def dataloader(self):
        if self.cfg['data_new'] == 1:
            train_data = Subset(self.dataset, indices=self.traintest_index[1])
            test_data = Subset(self.dataset, indices=self.traintest_index[2])
            # train_size = int(self.cfg['train_rate'] * len(train_data))
            valid_size = int(self.cfg['verify_rate'] * len(test_data))
            test_size = len(test_data) - valid_size
            test_dataset, valid_dataset = torch.utils.data.random_split(test_data, [test_size, valid_size])

            self.train_loader = DataLoader(dataset=train_data, batch_size=self.cfg['batchsize'], shuffle=True,
                                           num_workers=self.num_workers)
            self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                          num_workers=self.num_workers)
            self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.cfg['color_batchsize'], shuffle=False,
                                           num_workers=self.num_workers)

            color_data1 = Subset(self.dataset, indices=self.matrix_[1])
            self.color_loader1 = DataLoader(dataset=color_data1, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                            num_workers=self.num_workers)
            color_data2 = Subset(self.dataset, indices=self.matrix_[0])
            self.color_loader2 = DataLoader(dataset=color_data2, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                            num_workers=self.num_workers)
        else:
            train_data = Subset(self.dataset, indices=self.matrix_[1])
            train_size = int(self.cfg['train_rate'] * len(train_data))
            valid_size = int(self.cfg['verify_rate'] * len(train_data))
            test_size = len(train_data) - train_size - valid_size
            train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(train_data,
                                                                                       [train_size, test_size,
                                                                                        valid_size])

            self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg['batchsize'], shuffle=True,
                                           num_workers=self.num_workers)
            self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                          num_workers=self.num_workers)
            self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.cfg['color_batchsize'], shuffle=False,
                                           num_workers=self.num_workers)

            color_data = Subset(self.dataset, indices=self.matrix_[0])
            self.color_loader1 = DataLoader(dataset=train_data, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                            num_workers=self.num_workers)
            self.color_loader2 = DataLoader(dataset=color_data, batch_size=self.cfg['test_batchsize'], shuffle=False,
                                            num_workers=self.num_workers)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def indicator(self):
        savepath = self.cfg['RESULT_output'] + str(self.time) + "_matrix.npy" if self.cfg['test'][
            'save_matrix'] else None
        np.save(savepath, self.test_matrix)
        result = aa_oa(self.test_matrix)
        expo_result(result, self.cfg, [self.train_time, self.test_time], self.time)

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self):
        while self.time < self.TIME:
            self.train()
            # self.eval()
            # self.save_checkpoint()
            # self.save_records()
            self.time += 1
        # self.logger.log('Training done.')


# class BaseSolver:  # old
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.task = cfg['task']
#         self.TIME = cfg['time']
#         self.time = cfg['index']
#         self.EPOCH = cfg['epoch']
#         self.epoch = 0
#         self.DEVICE = cfg['device']
#
#         self.timestamp = int(time.time())
#
#         if cfg['gpu_mode']:
#             self.num_workers = cfg['threads']
#         else:
#             self.num_workers = 0
#
#         self.ms = read_tif(cfg, 'ms')
#         self.pan = read_tif(cfg, 'pan')
#
#         self.MS = data_padding(self.ms, cfg, 'ms')
#         self.PAN = data_padding(self.pan, cfg, 'pan')
#
#         if not os.path.exists(cfg['data_address']+'label.npy'):
#             label_mat2np(cfg)
#         label_np = np.load(cfg['data_address']+'label.npy', encoding='bytes', allow_pickle=True)
#         data_show(label_np)
#         xyl_matrix, self.matrix_ = split_data(label_np, cfg)
#
#         if cfg['use_h5']:
#             if not os.path.exists(cfg['data_address']+str(cfg['patch_size'])+"_train.h5"):
#                 dataset_cut(self.PAN, self.MS, xyl_matrix, self.matrix_[0], cfg, mode='train')
#                 dataset_cut(self.PAN, self.MS, xyl_matrix, self.matrix_[1], cfg, mode='color')
#             self.train_data = read_h5(cfg['data_address']+str(cfg['patch_size'])+"_train.h5")
#             print('Train dataset大小为:', self.train_data['ms'].shape)
#             self.dataset = dataset_h5(self.train_data)
#             print('All dataset大小为:', self.dataset.__len__())
#             if cfg['color']['index']:
#                 self.color_data = read_h5(cfg['data_address']+str(cfg['patch_size'])+"_train.h5")
#                 print('color dataset大小为:', self.color_data['ms'].shape)
#         else:
#             self.dataset = dataset_dual(self.MS, self.PAN, xyl_matrix, cfg)  # 总的数据集
#             print('All dataset大小为:', self.dataset.__len__())
#
#         self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}
#
#         # if not os.path.exists(self.checkpoint_dir):
#         #     os.makedirs(self.checkpoint_dir)
#
#     def dataloader(self):  # old
#         train_data = Subset(self.dataset, indices=self.matrix_[1])
#         train_size = int(self.cfg['train_rate'] * len(train_data))
#         valid_size = int(self.cfg['verify_rate'] * len(train_data))
#         test_size = len(train_data) - train_size - valid_size
#         train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(train_data,
#                                                                                    [train_size, test_size, valid_size])
#
#         self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg['batchsize'], shuffle=True, num_workers=self.num_workers)
#         self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.cfg['test_batchsize'], shuffle=False, num_workers=self.num_workers)
#         self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.cfg['color_batchsize'], shuffle=False, num_workers=self.num_workers)
#
#         color_data = Subset(self.dataset, indices=self.matrix_[0])
#         self.color_loader1 = DataLoader(dataset=train_data, batch_size=self.cfg['test_batchsize'], shuffle=False, num_workers=self.num_workers)
#         self.color_loader2 = DataLoader(dataset=color_data, batch_size=self.cfg['test_batchsize'], shuffle=False, num_workers=self.num_workers)
#
#     def load_checkpoint(self, model_path):
#         if os.path.exists(model_path):
#             ckpt = torch.load(model_path)
#             self.epoch = ckpt['epoch']
#             self.records = ckpt['records']
#         else:
#             raise FileNotFoundError
#
#     def save_checkpoint(self):
#         self.ckp = {
#             'epoch': self.epoch,
#             'records': self.records,
#         }
#
#     def indicator(self):
#         savepath = self.cfg['RESULT_output'] + str(self.time) + "_matrix.npy" if self.cfg['test']['save_matrix'] else None
#         np.save(savepath, self.test_matrix)
#         result = aa_oa(self.test_matrix)
#         expo_result(result, self.cfg, [self.train_time, self.test_time], self.time)
#
#     def train(self):
#         raise NotImplementedError
#
#     def eval(self):
#         raise NotImplementedError
#
#     def run(self):
#         while self.time < self.TIME:
#             self.train()
#             # self.eval()
#             # self.save_checkpoint()
#             # self.save_records()
#             self.time += 1
#         #self.logger.log('Training done.')

