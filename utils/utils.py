import os, torch, random
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from train.loss_function import qua_loss


def make_optimizer(cfg, params):
    opt_type = cfg['schedule']['optimizer']
    if opt_type == "ADAM":
        # optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer


def make_loss(loss_type, cfg):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='mean')
    elif loss_type == "L1":
        loss = nn.L1Loss(reduction='mean')
    elif loss_type == "Criterion":
        loss = nn.CrossEntropyLoss()
    elif loss_type == "KL":
        loss = nn.KLDivLoss(reduction='batchmean')
    elif loss_type == 'qua_loss':
        loss = qua_loss()
    else:
        raise ValueError
    return loss


def make_scheduler(optimizer, cfg):
    scheduler_type = cfg['schedule']['scheduler']
    if cfg['schedule']['if_scheduler']:
        if scheduler_type == "StepLR":  # 最简单的学习率调整方法
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50,
                                            gamma=cfg['schedule']['base_lr']/cfg['schedule']['lr'])
        elif scheduler_type == "LinearLR":  # 线性调整学习率
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=10)
        elif scheduler_type == "CosineAnnealingLR":  # 余弦退火学习率
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 50, cfg['schedule']['base_lr'])
        elif scheduler_type == "CyclicLR":  # 上坡下坡学习率
            scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=cfg['schedule']['base_lr'],
                                                          max_lr=cfg['schedule']['lr'], step_size_up=10,
                                                          step_size_down=40, cycle_momentum=False)
        elif scheduler_type == "OneCycleLR":  # CyclicLR的一周期版本
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['schedule']['lr'], pct_start=0.5,
                                                total_steps=cfg['epoch'],
                                                div_factor=cfg['schedule']['lr']/cfg['schedule']['base_lr'],
                                                final_div_factor=cfg['schedule']['lr']/cfg['schedule']['base_lr'])
        elif scheduler_type == "ConstantLR":  # 起始低学习率后续恢复
            scheduler = lr_scheduler.ConstantLR(optimizer, factor=cfg['schedule']['base_lr']/cfg['schedule']['lr'],
                                                total_iters=10)
        elif scheduler_type == "ChainedScheduler":  # 复合学习率
            scheduler = lr_scheduler.ChainedScheduler([lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                                             end_factor=1, total_iters=10),
                                                       lr_scheduler.ExponentialLR(optimizer, gamma=0.98)])
        elif scheduler_type == "ExponentialLR":
           scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
        else:
            raise ValueError
    else:
        return None
    return scheduler


def save_point_sche(model, optimizer, schedule, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
    }
    torch.save(checkpoint, filename)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # 预训练模型 要求模型一模一样，每一层的写法和命名都要一样  本质一样都不行
    # 完全一样的模型实现，也可能因为load和当初save时 pytorch版本不同 而导致state_dict中的key不一样
    # 例如 "initial.0.weight" 与 “initial.weight” 的区别
    model.load_state_dict(checkpoint["state_dict"], strict=False)   # 改成strict=False才能编译通过
    optimizer.load_state_dict(checkpoint["optimizer"])

    # 如果我们不这样做，将还会使用old checkpoint 中的 lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_model(checkpoint_file, model, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # 预训练模型 要求模型一模一样，每一层的写法和命名都要一样  本质一样都不行
    # 完全一样的模型实现，也可能因为load和当初save时 pytorch版本不同 而导致state_dict中的key不一样
    # 例如 "initial.0.weight" 与 “initial.weight” 的区别
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 改成strict=False才能编译通过


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False