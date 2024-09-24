import torch
import torch.nn as nn
import torch.nn.functional as F

deivce = "cuda" if torch.cuda.is_available() else "cpu"


def DL_divergence(p, q, r=0):
    p = torch.abs(p)
    q = torch.abs(q)
    result = p * torch.abs(torch.log(p / q))
    return torch.abs(result)


class qua_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def loss_qua(self, out, bs, cfg):
        data = out.softmax(dim=-1)
        epsilon = cfg['dqtl']['epsilon']
        p, q, r, s = data[:bs], data[bs:2 * bs], data[2 * bs:3 * bs], data[3 * bs:]# l_out = out[4*bs:]
        MSE = nn.MSELoss()
        smoothl1 = nn.SmoothL1Loss()
        # P, Q, R, S = data[:bs], data[bs:2 * bs], data[2 * bs:3 * bs], data[3 * bs:]

        # print(KL_M_P, KL_M_GM, torch.pow(KL_M_GP-KL_M_GM+tao, 2))
        # print(KL_P_M, KL_P_GP, torch.pow(KL_P_GM-KL_P_GP+tao, 2))
        # l1 = torch.pow(KL_M_P - KL_M_GM, 2) + KL_M_GM + torch.pow(KL_M_GP - KL_M_GM + tao, 2)
        # l2 = torch.pow(KL_P_M - KL_P_GP, 2) + KL_P_GP + torch.pow(KL_P_GM - KL_P_GP + tao, 2)
        # l3 = KL_M_GP / p + KL_P_GM / q
        # print(l1, l2, l3)
        # l1 = torch.exp(-torch.clip(KL_M_P + KL_M_GM - Kl_M_GP, min=0))
        # l2 = torch.exp(-torch.clip(KL_P_M + KL_P_GP - KL_P_GM, min=0))
        KL_M_P = F.kl_div((q + epsilon).log(), p, reduction='batchmean')
        KL_M_GM = F.kl_div((r + epsilon).log(), p, reduction='batchmean')
        KL_M_GP = F.kl_div((s + epsilon).log(), p, reduction='batchmean')
        KL_P_M = F.kl_div((p + epsilon).log(), q, reduction='batchmean')
        KL_P_GP = F.kl_div((r + epsilon).log(), q, reduction='batchmean')
        KL_P_GM = F.kl_div((s + epsilon).log(), q, reduction='batchmean')
        tao = cfg['dqtl']['tao']
        l1 = KL_M_P + KL_M_GM + torch.abs(KL_M_GP - KL_M_GM + tao)
        l2 = KL_P_M + KL_P_GP + torch.abs(KL_P_GM - KL_P_GP + tao)
        return l1, l2

    def loss_bal(self, out, bs, cfg):
        data = out.softmax(dim=-1)
        p, q, r, s = data[:bs], data[bs:2 * bs], data[2 * bs:3 * bs], data[3 * bs:]
        epsilon = cfg['dqtl']['epsilon']
        KL_M_GP = F.kl_div((s + epsilon).log(), p, reduction='batchmean')
        KL_P_GM = F.kl_div((s + epsilon).log(), q, reduction='batchmean')
        l3 = torch.mean(torch.exp(-torch.abs(KL_M_GP / p)) + torch.exp(-torch.abs(KL_P_GM / q)))
        return l3

    def loss_class(self, out, bs, t, cfg):
        data = out.softmax(dim=-1)
        p, q, r, s = data[:bs], data[bs:2 * bs], data[2 * bs:3 * bs], data[3 * bs:]
        label = torch.zeros(p.shape).to(cfg['device'])
        for i in range(label.shape[0]):
            label[i][int(t[i])] = 1
        l = label.softmax(dim=-1)
        l4 = F.kl_div((p+q).softmax(dim=-1).log(), l, reduction='batchmean')
        return l4


    def forward(self, out, bs, t, cfg):
        # label = torch.zeros(p.shape).to(cfg['device'])
        # for i in range(label.shape[0]):
        #     label[i][int(t[i])] = 1

        alpha = cfg['dqtl']['alpha']
        beta = cfg['dqtl']['beta']
        gamma = cfg['dqtl']['gamma']
        if alpha != 0:
            l1, l2 = self.loss_qua(out, bs, cfg)
        else:
            l1, l2 = 0, 0
        if beta != 0:
            l3 = self.loss_bal(out, bs, cfg)
        else:
            l3 = 0
        l4 = self.loss_class(out, bs, t, cfg)

        loss = alpha * (l1 + l2) + beta * l3 + gamma * l4
        return loss


# test()
if __name__ == '__main__':
    loss = qua_loss()
    cfg = {
        "device": 'cuda',
        "dqtl":{"alpha": 0.1,
                "beta": 0,
                "gamma": 1.0
        }
    }
    x = torch.randn(40, 8).to('cuda')
    l = torch.randint(1, 8, (10, 1)).squeeze()
    # print(l)
    y = loss(x, 10, l.to('cuda'), cfg)
    print(y)
import numpy as np
import scipy.stats

# def KL(p,q):
#     return F.kl_div(p, q, reduction='batchmean')
#
# p=torch.tensor([0.01,0.01,0.9,0.1])
# q=torch.tensor([0.6,0.1,0.1,10])
# z=torch.tensor([0.61,0.26,0.12,10])
# print(p.softmax(dim=-1), p.softmax(dim=-1).log(), q.softmax(dim=-1))
# print(KL(q.softmax(dim=-1).log(), q.softmax(dim=-1))) # 0.011735745199107783
# print(KL(q.softmax(dim=-1).log(), p.softmax(dim=-1))) # 0.013183150978050884
# print(KL(q.softmax(dim=-1).log(), z.softmax(dim=-1))) # 0
