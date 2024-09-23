import torch
import torch.nn as nn
from model.panform.panformer import CrossSwinTransformer

def Net(args):
    return CrossSwinTransformer(num_classes=args['Categories_Number'])


if __name__ == '__main__':
    device = 'cuda:0'
    ms = torch.randn([20, 4, 16, 16]).to(device)
    pan = torch.randn([20, 1, 64, 64]).to(device)
    cfg = {
        'Categories_Number': 10,
        'device': 'cuda:0'
    }
    model = Net(cfg).to(device)
    y = model(ms, pan)
    print(y.shape)