import torch
import numpy as np
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lp_filters(c):
    h = torch.tensor([.037828455506995, -.023849465019380,
                      -.11062440441842, .37740285561265, .85269867900940,
                      .37740285561265, -.11062440441842, -.023849465019380,
                      .037828455506995]).to(device)
    h = torch.unsqueeze(h, 1)
    h = h*h.T.unsqueeze(0)  # torch.Size([1, 9, 9])
    g = torch.tensor([-.064538882628938, -.040689417609558,
                      .41809227322221, .78848561640566,
                      .41809227322221, -.040689417609558,
                      -.064538882628938]).to(device)
    g = torch.unsqueeze(g, 1)
    g = g*g.T.unsqueeze(0)  # torch.Size([1, 7, 7])
    return h.expand(c, 1, 9, 9), g.expand(c, 1, 7, 7)


def alpha_filter(c, a):
    h = torch.tensor([0.25-a/2, 0.25, a, 0.25, 0.25-a/2])
    h = torch.unsqueeze(h, 1)
    h = h * h.T.unsqueeze(0)  # torch.Size([1, 5, 5])

    g = torch.tensor([-.064538882628938, -.040689417609558,
                      .41809227322221, .78848561640566,
                      .41809227322221, -.040689417609558,
                      -.064538882628938])
    g = torch.unsqueeze(g, 1)
    g = g * g.T.unsqueeze(0)  # torch.Size([1, 7, 7])
    return h.expand(c, 1, 5, 5).to(device), g.expand(c, 1, 7, 7).to(device)


def dfb_filters(c, mode=None, name=None):
    if name == 'haar':
        if mode == 'r':
            g0 = torch.tensor([[1, 1]])/torch.sqrt(torch.tensor([2]))
            g0 = g0.expand(c, 1, 1, 2)
            g1 = torch.tensor([[1, -1]])/torch.sqrt(torch.tensor([2]))
            g1 = g1.expand(c, 1, 1, 2)
            return g0.to(device), g1.to(device)
        elif mode == 'd':
            h0 = torch.tensor([[1, 1]])/torch.sqrt(torch.tensor([2]))
            h0 = h0.expand(c, 1, 1, 2)
            h1 = torch.tensor([[-1, 1]])/torch.sqrt(torch.tensor([2]))
            h1 = h1.expand(c, 1, 1, 2)
            return h0.to(device), h1.to(device)
        else:
            raise NotImplementedError("Mode is not available")
    elif name == 'thanh':
        if mode == 'r':
            g0 = - torch.tensor([[0, -1, 0],
                             [-1, -4, -1],
                             [0, -1, 0]]) / 4.0
            g0 = g0.expand(c, 1, 3, 3)
            g1 = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, -1, 0, 0],
                           [0, 0, 0, -2, -4, -2, 0],
                           [0, 0, -1, -4, 28, -4, -1],
                           [0, 0, 0, -2, -4, -2, 0],
                           [0, 0, 0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]])/32.0
            g1 = g1.expand(c, 1, 7, 7)
            return g0.to(device), g1.to(device)
        elif mode == 'd':
            h0 = torch.tensor([[0, 0, -1, 0, 0],
                           [0, -2, 4, -2, 0],
                           [-1, 4, 28, 4, -1],
                           [0, -2, 4, -2, 0],
                           [0, 0, -1, 0, 0]])/32.0
            h0 = h0.expand(c, 1, 5, 5)

            h1 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, -1, 0, 0, 0],
                           [-1, 4, -1, 0, 0],
                           [0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0]])/4.0
            h1 = h1.expand(c, 1, 5, 5)
            return h0.to(device), h1.to(device)
        else:
            raise NotImplementedError("Mode is not available")
    else:
        raise NotImplementedError("Filters haven't implemented")


if __name__ == '__main__':
    h, g = lp_filters(4)
    print('9-7 laplacian pyramid filters: ')
    print('h shape: ', h.shape)
    print(h)
    print('g shape: ', g.shape)
    print(g)

    mode = 'r'
    name = 'thanh'
    h0, h1 = dfb_filters(4, mode = mode, name=name)
    print('DFB filters')
    print('mode decompose' if mode=='r' else 'mode recompose')
    print('haar filters' if name=='haar' else 'thanh filters')
    print('h0 shape: ', h0.shape)
    print(h0)
    print('h1 shape: ', h1.shape)
    print(h1)