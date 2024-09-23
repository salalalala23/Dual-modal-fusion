import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
from torchvision import transforms


class ContourDec(nn.Module):
    def __init__(self, in_channels, nlevs, device='cuda'):
        super().__init__()
        nlevs = nlevs if type(nlevs) != int else [nlevs]
        self.nlevs = nlevs
        self.time = len(nlevs)
        self.device = device
        self.lap_filter()
        self.conv_xlo = self.conv_(in_channels, self.h, self.h)
        self.conv_d = self.conv_(in_channels, self.g, self.g)
        self.dfb_filter()
        self.conv_f = [self.conv_(in_channels*(i+1), self.f, self.f) for i in range(nlevs[0])]
        # print(self.conv_f)

    def conv_(self, ch, f1, f2):
        f1_len = len(f1)
        f2_len = len(f2)
        conv1 = nn.Conv2d(ch, ch, (f1_len, f2_len), 1)
        conv1.weight = nn.Parameter(torch.ger(f1, f2).repeat(ch, ch, 1, 1), requires_grad=False)
        return conv1.to(self.device)

    def lap_filter(self, fname="9-7"):
        if fname == "9-7":
            h = np.array([0.037828455506995, -0.02384946501938, -0.11062440441842, 0.37740285561265], dtype=np.float32)
            h = np.concatenate((h, [0.8526986790094], h[::-1]))

            g = np.array([-0.064538882628938, -0.040689417609558, 0.41809227322221], dtype=np.float32)
            g = np.concatenate((g, [0.78848561640566], g[::-1]))
        else:
            raise ValueError("the name of lpfilters is wrong")
        h = h.astype(np.float32)
        g = g.astype(np.float32)
        self.h, self.g = torch.from_numpy(h), torch.from_numpy(g)

    def dfb_filter(self):
        # length 12 filter from Phoong, Kim, Vaidyanathan and Ansari
        v = np.array([0.63, -0.193, 0.0972, -0.0526, 0.0272, -0.0144], dtype=np.float32)
        # Symmetric impulse response
        f = np.concatenate((v[::-1], v)).astype(np.float32)
        # Modulate f
        f[::2] = -f[::2]
        self.f = torch.from_numpy(f)

    def lpdec_layer(self, x):
        # Lowpass filter and downsample

        xlo = self.sefilter2_layer(x, self.h, self.h, self.conv_xlo, 'per')
        c = F.avg_pool2d(xlo, 2)

        # Compute the residual (bandpass) image by upsample, filter, and subtract
        # Even size filter needs to be adjusted to obtain perfect reconstruction
        adjust = (len(self.g) + 1) % 2
        # d = insert_zero(xlo) # d = Lambda(insert_zero)(xlo)
        d = dup(c)  # d = Lambda(insert_zero)(xlo)
        d = self.sefilter2_layer(d, self.g, self.g, self.conv_d, 'per', adjust * np.array([1, 1], dtype=np.float32))
        d = x - d  # d = Subtract()([x,d])
        return c, d

    def sefilter2_layer(self, x, f1, f2, conv, extmod='per', shift=[0, 0], device='cuda'):
        f1_len = len(f1)
        f2_len = len(f2)
        lf1 = (f1_len - 1) / 2
        lf2 = (f2_len - 1) / 2
        y = extend2_layer(x, int(np.floor(lf1) + shift[0]), int(np.ceil(lf1) - shift[0]),
                          int(np.floor(lf2) + shift[1]), int(np.ceil(lf2) - shift[1]), extmod)
        t1 = time()
        y = conv(y)
        # y = conv2(y, f3, device)
        # y = conv2(y, f4, device)
        t2 = time()
        print("conv2", t2 - t1)
        return y

    def dfbdec_layer(self, x):

        if self.nlevs[0] == 1:
            y = [None] * 2
            # Simplest case, one level
            y[0], y[1] = self.fbdec_layer(x, self.f, self.conv_f[0], 'q', '1r', 'qper_col')
        elif self.nlevs[0] >= 2:
            y = [None] * 4
            t1 = time()
            x0, x1 = self.fbdec_layer(x, self.f, self.conv_f[0], 'q', '1r', 'qper_col')
            t2 = time()
            print("fbdec_layer:::::", t2 - t1)
            # y[1], y[0] = fbdec_layer(x0, f, 'q', '2c', 'per')
            # y[3], y[2] = fbdec_layer(x1, f, 'q', '2c', 'per')
            odd_list, even_list = self.new_fbdec_layer([x0, x1], self.f, self.conv_f[1], 'q', '2c', 'per')
            t3 = time()
            print("new_fbdec:::::", t3 - t2)
            # y[1], y[2] = odd_list
            # y[0], y[3] = even_list
            for ix in range(len(odd_list)):
                y[ix * 2 + 1], y[ix * 2] = odd_list[ix], even_list[ix]

            # Now expand the rest of the tree
            for l in range(3, self.nlevs[0] + 1):
                # Allocate space for the new subband outputs
                y_old = y.copy()
                y = [None] * (2 ** l)

                # The first half channels use R1 and R2
                # for k in range(1, 2 ** (l - 2)+1):
                #     i = (k - 1) % 2 + 1
                #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
                odd = np.arange(1, 2 ** (l - 2) + 1, 2)
                even = np.arange(2, 2 ** (l - 2) + 1, 2)
                odd_list, even_list = self.new_fbdec_layer([y_old[k - 1] for k in odd], self.f, self.conv_f[l-1], 'p', 1, 'per')
                for ix, k in enumerate(odd):
                    y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]
                odd_list, even_list = self.new_fbdec_layer([y_old[k - 1] for k in even], self.f, self.conv_f[l-1], 'p', 2, 'per')
                for ix, k in enumerate(even):
                    y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]

                # The second half channels use R3 and R4
                # for k in range(2 ** (l - 2) + 1,2 ** (l - 1) + 1):
                #     i = (k - 1) % 2 + 3
                #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
                odd += 2 ** (l - 2)
                even += 2 ** (l - 2)
                odd_list, even_list = self.new_fbdec_layer([y_old[k - 1] for k in odd], self.f, 'p', 3, 'per')
                for ix, k in enumerate(odd):
                    y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]
                odd_list, even_list = new_fbdec_layer([y_old[k - 1] for k in even], self.f, 'p', 4, 'per')
                for ix, k in enumerate(even):
                    y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]
        t4 = time()
        print("odd even :::::", t4 - t3)

        # Backsampling
        def backsamp(y=None):
            n = np.log2(len(y))

            assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
            n = int(n)
            if n == 1:
                # One level, the decomposition filterbank shoud be Q1r
                # Undo the last resampling (Q1r = R2 * D1 * R3)
                for k in range(2):
                    y[k] = resamp(y[k], 4)
                    y[k][..., ::2] = resamp(y[k][..., ::2], 1)
                    y[k][..., 1::2] = resamp(y[k][..., 1::2], 1)

            if n > 2:
                N = 2 ** (n - 1)

                for k in range(1, 2 ** (n - 2) + 1):
                    shift = 2 * k - (2 ** (n - 2) + 1)

                    # The first half channels
                    # y[2*k - 2]=resamp(y[2*k - 2],3,shift)
                    # y[2*k - 1]=resamp(y[2*k - 1],3,shift)
                    y[2 * k - 2], y[2 * k - 1] = new_resamp([y[2 * k - 2], y[2 * k - 1]], 3, shift)

                    # The second half channels
                    # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,shift)
                    # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,shift)
                    y[2 * k - 2 + N], y[2 * k - 1 + N] = new_resamp([y[2 * k - 2 + N], y[2 * k - 1 + N]], 1, shift)

            return y

        y = backsamp(y)

        # Flip the order of the second half channels
        y[2 ** (self.nlevs[0] - 1):] = y[-1:2 ** (self.nlevs[0] - 1) - 1:-1]
        t5 = time()
        print("backsamp:::::", t5 - t4)
        return y

    def fbdec_layer(self, x, f_, conv, type1, type2, extmod='per'):
        # Polyphase decomposition of the input image
        t1 = time()
        if type1 == 'q':
            # Quincunx polyphase decomposition
            p0, p1 = self.qpdec_layer(x, type2)

        elif type1 == 'p':
            # Parallelogram polyphase decomposition
            p0, p1 = self.ppdec_layer(x, type2)
        else:
            raise AttributeError("type error")
        t2 = time()
        print("---select type", t2 - t1)
        # # Ladder network structure
        y0 = 1 / (2 ** 0.5) * (p0 - self.sefilter2_layer(p1, f_, f_, conv, extmod, [1, 1]))
        y1 = (-2 ** 0.5) * p1 - self.sefilter2_layer(y0, f_, f_, conv, extmod)
        print("---fbdec_conv", time() - t2)
        return [y0, y1]

    def new_fbdec_layer(self, x, f_, conv, type1, type2, extmod='per'):
        sample = len(x)
        # x = tf.concat(x, axis=-1)
        x = torch.cat(x, dim=1)
        ch = x.shape[1] // sample
        t1 = time()
        # Polyphase decomposition of the input image
        if type1 == 'q':
            # Quincunx polyphase decomposition
            p0, p1 = self.qpdec_layer(x, type2)

        elif type1 == 'p':
            # Parallelogram polyphase decomposition
            p0, p1 = self.ppdec_layer(x, type2)
        t2 = time()
        print("---type:", t2 - t1)

        # Ladder network structure
        y0 = 1 / (2 ** 0.5) * (p0 - self.sefilter2_layer(p1, f_, f_, conv, extmod, [1, 1]))
        y1 = (-2 ** 0.5) * p1 - self.sefilter2_layer(y0, f_, f_, conv, extmod)
        t3 = time()
        print("---y0y1", t3 - t2)
        # return [y0, y1]
        return [y0[:, i * ch:(i + 1) * ch] for i in range(sample)], [y1[:, i * ch:(i + 1) * ch] for i in range(sample)]

    def ppdec_layer(self, x, type_):
        # TODO
        if type_ == 1:  # P1 = R1 * Q1 = D1 * R3
            # p0=resamp(x[:,::2,:,:],3)

            # R1 * [0; 1] = [1; 1]
            # p1=resamp(np.roll(x[1::2,:],-1,axis=1),3)
            p1 = torch.cat([x[..., 1::2, 1:], x[..., 1::2, 0:1]], dim=3)
            # p1=resamp(p1, 3)
            # p0, p1 = new_resamp([x[:,::2,:,:], p1], 3)
            p0, p1 = new_resamp([x[..., ::2, :], p1], 3)

        elif type_ == 2:  # P2 = R2 * Q2 = D1 * R4
            # p0=resamp(x[:,::2,:,:],4)

            # R2 * [1; 0] = [1; 0]
            # p1=resamp(x[:,1::2,:,:],4)
            # p0, p1 = new_resamp([x[:,::2,:,:], x[:,1::2,:,:]], 4)
            p0, p1 = new_resamp([x[..., ::2, :], x[..., 1::2, :]], 4)

        elif type_ == 3:  # P3 = R3 * Q2 = D2 * R1
            # p0=resamp(x[:,:,::2,:],1)

            # R3 * [1; 0] = [1; 1]
            # p1=resamp(np.roll(x[:,1::2],-1,axis=0),1)
            # p1 = torch.cat([x[:,1:,1::2,:], x[:,0:1,1::2,:]], dim=1)
            p1 = torch.cat([x[..., 1:, 1::2], x[..., 0:1, 1::2]], dim=2)
            # p1=resamp(p1, 1)
            # p0, p1 = new_resamp([x[:,:,::2,:], p1], 1)
            p0, p1 = new_resamp([x[..., ::2], p1], 1)

        elif type_ == 4:  # P4 = R4 * Q1 = D2 * R2
            # p0=resamp(x[:,:,::2,:],2)

            # R4 * [0; 1] = [0; 1]
            # p1=resamp(x[:,:,1::2,:],2)
            # p0, p1 = new_resamp([x[:,:,::2,:], x[:,:,1::2,:]], 2)
            p0, p1 = new_resamp([x[..., ::2], x[..., 1::2]], 2)

        else:
            raise ValueError('Invalid argument type')

        return p0, p1

    def qpdec_layer(self, x, type_='1r'):
        if type_ == '1r':  # Q1 = R2 * D1 * R3
            t1 = time()
            y = resamp(x, 2)
            t2 = time()
            # print(t2 - t1)
            # p0 = resamp(y[:,::2,:,:], 3)

            # inv(R2) * [0; 1] = [1; 1]
            # p1 = resamp(y(2:2:end, [2:end, 1]), 3)
            p1 = torch.cat([y[..., 1::2, 1:], y[..., 1::2, 0:1]], dim=3)
            # p1 = resamp(p1, 3)
            p0, p1 = new_resamp([y[..., ::2, :], p1], 3)
            t3 = time()
            # print(t3 - t2)
        elif type_ == '1c':  # Q1 = R3 * D2 * R2
            # TODO
            y = resamp(x, 3)

            # p0=resamp(y[:,:,::2,:],2)
            p0 = resamp(y[..., ::2], 2)

            # inv(R3) * [0; 1] = [0; 1]
            # p1=resamp(y[:,:,1::2,:],2)
            p1 = resamp(y[..., 1::2], 2)

        elif type_ == '2r':  # Q2 = R1 * D1 * R4
            # TODO
            y = resamp(x, 1)

            p0 = resamp(y[..., ::2, :], 4)

            # inv(R1) * [1; 0] = [1; 0]
            p1 = resamp(y[..., 1::2, :], 4)

        elif type_ == '2c':  # Q2 = R4 * D2 * R1
            y = resamp(x, 4)

            # p0=resamp(y[:,:,::2,:],1)

            # inv(R4) * [1; 0] = [1; 1]
            # p1 = resamp(y([2:end, 1], 2:2:end), 1)
            p1 = torch.cat([y[..., 1:, 1::2], y[..., 0:1, 1::2]], dim=2)
            # p1 = resamp(p1,1)
            # p0, p1 = new_resamp([y[:,:,::2,:], p1], 1)
            p0, p1 = new_resamp([y[..., ::2], p1], 1)

        else:
            raise ValueError('Invalid argument type')

        return p0, p1

    def forward(self, x, pfilt=None, dfilt=None):
        t1 = time()
        print(x.shape)
        lowpass, bandpass = [], []
        for i in range(self.time):
            if self.nlevs != 0:
                # Laplacian decomposition
                t1 = time()
                xlo, xhi = self.lpdec_layer(x)
                t2 = time()
                print("------\nlpdec", t2 - t1)
                # print(xlo.shape, xhi.shape)

                # Use the ladder structure (which is much more efficient)
                xhi = self.dfbdec_layer(xhi)
                t3 = time()
                print("------\ndfbdec", t3 - t2)
                for i in range(len(xhi)):
                    k = xhi[i].shape
                    if k[2] != k[3]:
                        # Get maximum dimension (height or width)
                        max_dim = int(np.max((k[2], k[3])))
                        # Resize the channels
                        trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                        xhi[i] = trans(xhi[i])
                xhi = torch.cat(xhi, dim=1)
                lowpass.append(xlo)
                bandpass.append(xhi)
                x = xlo
        return lowpass,bandpass
        # return pdfbdec_layer(x, self.nlevs, self.device)


def pdfbdec_layer(x, nlevs, device='cuda', pfilt=None, dfilt=None):
    t1 = time()
    if nlevs != 0:
        # Laplacian decomposition
        t1 = time()
        xlo, xhi = lpdec_layer(x, device)
        t2 = time()
        print("lpdec", t2 - t1)
        # print(xlo.shape, xhi.shape)

        # Use the ladder structure (which is much more efficient)
        xhi = dfbdec_layer(xhi, dfilt, nlevs)
        t3 = time()
        #print("dfbdec", t3 - t2)
        for i in range(len(xhi)):
            k = xhi[i].shape
            if k[2] != k[3]:
                # Get maximum dimension (height or width)
                max_dim = int(np.max((k[2], k[3])))
                # Resize the channels
                trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                xhi[i] = trans(xhi[i])
        xhi = torch.cat(xhi, dim=1)
    return xlo, xhi


def lpdec_layer(x, device='cuda'):
    h, g = lap_filter(device, x.dtype)
    # ch = x.shape[1]
    # h_l, h_p = len(h), int(len(h)/2)
    # conv1 = nn.Conv2d(ch, ch, h_l, 1).to(device)
    # conv1.weight = nn.Parameter(torch.ger(h, h).repeat(ch, ch, 1, 1), requires_grad= False)
    # x_padded = F.pad(x, pad=(h_p, h_p, h_p, h_p), mode='reflect')
    # xlo = conv1(x_padded)

    # Lowpass filter and downsample
    xlo = sefilter2_layer(x, h, h, 'per', device=device)

    # c = xlo[:, :, ::2, ::2]
    c = F.avg_pool2d(xlo, 2)

    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = (len(g) + 1) % 2

    # d = insert_zero(xlo) # d = Lambda(insert_zero)(xlo)
    d = dup(c)  # d = Lambda(insert_zero)(xlo)
    # d = F.interpolate(c, scale_factor=[2, 2], mode='nearest')
    # g_l, g_p = len(g), int(len(g)/2)
    # conv2 = nn.Conv2d(ch, ch, g_l, 1).to(device)
    # conv2.weight = nn.Parameter(torch.ger(g, g).repeat(ch, ch, 1, 1), requires_grad= False)
    # d_padded = F.pad(d, pad=(g_p, g_p, g_p, g_p), mode='reflect')
    # d = conv2(d_padded)
    d = sefilter2_layer(d, g, g, 'per', adjust * np.array([1, 1], dtype=np.float32), device)

    d = x - d  # d = Subtract()([x,d])
    return c, d


def lap_filter(device, dtype, fname="9-7"):
    if fname == "9-7":
        h = np.array([0.037828455506995, -0.02384946501938, -0.11062440441842, 0.37740285561265], dtype=np.float32)
        h = np.concatenate((h, [0.8526986790094], h[::-1]))

        g = np.array([-0.064538882628938, -0.040689417609558, 0.41809227322221], dtype=np.float32)
        g = np.concatenate((g, [0.78848561640566], g[::-1]))
    else:
        raise ValueError("the name of lpfilters is wrong")
    # h = np.outer(h, h).reshape(1, 1, 9, 9)
    # g = np.outer(g, g).reshape(1, 1, 7, 7)

    h, g = torch.from_numpy(h).to(dtype), torch.from_numpy(g).to(dtype)
    h, g = h.to(device), g.to(device)
    return h, g


def conv2(x, W, device='cuda', C=1, strides=[1, 1, 1, 1], padding=0):
    return F.conv2d(x.to(device), W.to(device), padding=padding, groups=C)


def extend2_layer(x, ru, rd, cl, cr, extmod):
    x = x
    # rx, cx = x.get_shape().as_list()[1:3]
    rx, cx = x.shape[2:]
    if extmod == 'per':

        # y = torch.cat([x[..., rx-ru:rx,:],x,x[..., :rd,:]], dim=2)
        # y = torch.cat([y[..., cx-cl:cx],y,y[..., :cr]], dim=3)

        y = torch.cat([x[..., -ru:, :], x, x[..., :rd, :]], dim=2)
        y = torch.cat([y[..., -cl:], y, y[..., :cr]], dim=3)

    elif extmod == 'qper_row':
        raise ValueError
        rx2 = round(rx / 2)
        y1 = K.concatenate([x[:, rx2:rx, cx - cl:cx, :], x[:, :rx2, cx - cl:cx, :]], axis=1)
        y2 = K.concatenate([x[:, rx2:rx, :cr, :], x[:, :rx2, :cr, :]], axis=1)
        y = K.concatenate([y1, x, y2], axis=1)

        y = K.concatenate([y[:, rx - ru:rx, :, :], y, y[:, :rd, :, :]], axis=1)

    elif extmod == 'qper_col':

        cx2 = round(cx / 2)
        y1 = torch.cat([x[..., rx - ru:rx, cx2:cx], x[..., rx - ru:rx, :cx2]], dim=3)
        y2 = torch.cat([x[..., :rd, cx2:cx], x[..., :rd, :cx2]], dim=3)
        y = torch.cat([y1, x, y2], dim=2)

        y = torch.cat([y[..., cx - cl:cx], y, y[..., :cr]], dim=3)
    return y


def sefilter2_layer(x, f1, f2, extmod='per',shift=[0,0], device='cuda'):
    f1_len = len(f1)
    f2_len = len(f2)
    lf1 = (f1_len - 1) / 2
    lf2 = (f2_len - 1) / 2
    y = extend2_layer(x, int(np.floor(lf1) + shift[0]), int(np.ceil(lf1) - shift[0]), \
                      int(np.floor(lf2) + shift[1]), int(np.ceil(lf2) - shift[1]), extmod)

    ch = y.shape[1]
    conv1 = nn.Conv2d(ch, ch, (f1_len, f2_len), 1).to(device)
    conv1.weight = nn.Parameter(torch.ger(f1, f2).repeat(ch, ch, 1, 1), requires_grad=False)
    t1 = time()
    y = conv1(y)
    # y = conv2(y, f3, device)
    # y = conv2(y, f4, device)
    t2 = time()
    # #print("conv2", t2 - t1)
    return y


def dup(x, step=[2,2]):
    N,C,H,W = x.shape
    y = torch.zeros((N,C,H*step[0],W*step[1]), device=x.device)
    y[...,::step[0],::step[1]]=x
    return y


def dfbdec_layer(x, f, n, device='cuda'):
    f = dfb_filter(x.device)

    if n == 1:
        y = [None] * 2
        # Simplest case, one level
        y[0], y[1] = fbdec_layer(x, f, 'q', '1r', 'qper_col')
    elif n >= 2:
        y = [None] * 4
        x0, x1 = fbdec_layer(x, f, 'q', '1r', 'qper_col')
        # y[1], y[0] = fbdec_layer(x0, f, 'q', '2c', 'per')
        # y[3], y[2] = fbdec_layer(x1, f, 'q', '2c', 'per')
        odd_list, even_list = new_fbdec_layer([x0, x1], f, 'q', '2c', 'per')
        # y[1], y[2] = odd_list
        # y[0], y[3] = even_list
        for ix in range(len(odd_list)):
            y[ix * 2 + 1], y[ix * 2] = odd_list[ix], even_list[ix]

        # Now expand the rest of the tree
        for l in range(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y.copy()
            y = [None] * (2 ** l)

            # The first half channels use R1 and R2
            # for k in range(1, 2 ** (l - 2)+1):
            #     i = (k - 1) % 2 + 1
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd = np.arange(1, 2 ** (l - 2) + 1, 2)
            even = np.arange(2, 2 ** (l - 2) + 1, 2)
            odd_list, even_list = new_fbdec_layer([y_old[k - 1] for k in odd], f, 'p', 1, 'per')
            for ix, k in enumerate(odd):
                y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k - 1] for k in even], f, 'p', 2, 'per')
            for ix, k in enumerate(even):
                y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]

            # The second half channels use R3 and R4
            # for k in range(2 ** (l - 2) + 1,2 ** (l - 1) + 1):
            #     i = (k - 1) % 2 + 3
            #     y[2*k-1], y[2*k-2] = fbdec_layer(y_old[k-1], f, 'p', i, 'per')
            odd += 2 ** (l - 2)
            even += 2 ** (l - 2)
            odd_list, even_list = new_fbdec_layer([y_old[k - 1] for k in odd], f, 'p', 3, 'per')
            for ix, k in enumerate(odd):
                y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]
            odd_list, even_list = new_fbdec_layer([y_old[k - 1] for k in even], f, 'p', 4, 'per')
            for ix, k in enumerate(even):
                y[2 * k - 1], y[2 * k - 2] = odd_list[ix], even_list[ix]

    # Backsampling
    def backsamp(y=None):
        n = np.log2(len(y))

        assert not (n != round(n) or n < 1), 'Input must be a cell vector of dyadic length'
        n = int(n)
        if n == 1:
            # One level, the decomposition filterbank shoud be Q1r
            # Undo the last resampling (Q1r = R2 * D1 * R3)
            for k in range(2):
                y[k] = resamp(y[k], 4)
                y[k][..., ::2] = resamp(y[k][..., ::2], 1)
                y[k][..., 1::2] = resamp(y[k][..., 1::2], 1)

        if n > 2:
            N = 2 ** (n - 1)

            for k in range(1, 2 ** (n - 2) + 1):
                shift = 2 * k - (2 ** (n - 2) + 1)

                # The first half channels
                # y[2*k - 2]=resamp(y[2*k - 2],3,shift)
                # y[2*k - 1]=resamp(y[2*k - 1],3,shift)
                y[2 * k - 2], y[2 * k - 1] = new_resamp([y[2 * k - 2], y[2 * k - 1]], 3, shift)

                # The second half channels
                # y[2*k - 2 + N]=resamp(y[2*k - 2 + N],1,shift)
                # y[2*k - 1 + N]=resamp(y[2*k - 1 + N],1,shift)
                y[2 * k - 2 + N], y[2 * k - 1 + N] = new_resamp([y[2 * k - 2 + N], y[2 * k - 1 + N]], 1, shift)

        return y

    y = backsamp(y)

    # Flip the order of the second half channels
    y[2 ** (n - 1):] = y[-1:2 ** (n - 1) - 1:-1]

    return y


def fbdec_layer(x, f_, type1, type2, extmod='per'):
    # Polyphase decomposition of the input image
    ch = x.shape[1]
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0, p1 = qpdec_layer(x, type2)

    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0, p1 = ppdec_layer(x, type2)
    else:
        raise AttributeError("type error")

    # # Ladder network structure
    y0 = 1 / (2 ** 0.5) * (p0 - sefilter2_layer(p1, f_, f_, extmod, [1, 1]))
    y1 = (-2 ** 0.5) * p1 - sefilter2_layer(y0, f_, f_, extmod)

    return [y0, y1]


def qpdec_layer(x, type_='1r'):
    if type_ == '1r':  # Q1 = R2 * D1 * R3
        y = resamp(x, 2)

        # p0 = resamp(y[:,::2,:,:], 3)

        # inv(R2) * [0; 1] = [1; 1]
        # p1 = resamp(y(2:2:end, [2:end, 1]), 3)
        p1 = torch.cat([y[..., 1::2, 1:], y[..., 1::2, 0:1]], dim=3)
        # p1 = resamp(p1, 3)
        p0, p1 = new_resamp([y[..., ::2, :], p1], 3)

    elif type_ == '1c':  # Q1 = R3 * D2 * R2
        # TODO
        y = resamp(x, 3)

        # p0=resamp(y[:,:,::2,:],2)
        p0 = resamp(y[..., ::2], 2)

        # inv(R3) * [0; 1] = [0; 1]
        # p1=resamp(y[:,:,1::2,:],2)
        p1 = resamp(y[..., 1::2], 2)

    elif type_ == '2r':  # Q2 = R1 * D1 * R4
        # TODO
        y = resamp(x, 1)

        p0 = resamp(y[..., ::2, :], 4)

        # inv(R1) * [1; 0] = [1; 0]
        p1 = resamp(y[..., 1::2, :], 4)

    elif type_ == '2c':  # Q2 = R4 * D2 * R1
        y = resamp(x, 4)

        # p0=resamp(y[:,:,::2,:],1)

        # inv(R4) * [1; 0] = [1; 1]
        # p1 = resamp(y([2:end, 1], 2:2:end), 1)
        p1 = torch.cat([y[..., 1:, 1::2], y[..., 0:1, 1::2]], dim=2)
        # p1 = resamp(p1,1)
        # p0, p1 = new_resamp([y[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([y[..., ::2], p1], 1)

    else:
        raise ValueError('Invalid argument type')

    return p0, p1


def ppdec_layer(x, type_):
    # TODO
    if type_ == 1:  # P1 = R1 * Q1 = D1 * R3
        # p0=resamp(x[:,::2,:,:],3)

        # R1 * [0; 1] = [1; 1]
        # p1=resamp(np.roll(x[1::2,:],-1,axis=1),3)
        p1 = torch.cat([x[..., 1::2, 1:], x[..., 1::2, 0:1]], dim=3)
        # p1=resamp(p1, 3)
        # p0, p1 = new_resamp([x[:,::2,:,:], p1], 3)
        p0, p1 = new_resamp([x[..., ::2, :], p1], 3)

    elif type_ == 2:  # P2 = R2 * Q2 = D1 * R4
        # p0=resamp(x[:,::2,:,:],4)

        # R2 * [1; 0] = [1; 0]
        # p1=resamp(x[:,1::2,:,:],4)
        # p0, p1 = new_resamp([x[:,::2,:,:], x[:,1::2,:,:]], 4)
        p0, p1 = new_resamp([x[..., ::2, :], x[..., 1::2, :]], 4)

    elif type_ == 3:  # P3 = R3 * Q2 = D2 * R1
        # p0=resamp(x[:,:,::2,:],1)

        # R3 * [1; 0] = [1; 1]
        # p1=resamp(np.roll(x[:,1::2],-1,axis=0),1)
        # p1 = torch.cat([x[:,1:,1::2,:], x[:,0:1,1::2,:]], dim=1)
        p1 = torch.cat([x[..., 1:, 1::2], x[..., 0:1, 1::2]], dim=2)
        # p1=resamp(p1, 1)
        # p0, p1 = new_resamp([x[:,:,::2,:], p1], 1)
        p0, p1 = new_resamp([x[..., ::2], p1], 1)

    elif type_ == 4:  # P4 = R4 * Q1 = D2 * R2
        # p0=resamp(x[:,:,::2,:],2)

        # R4 * [0; 1] = [0; 1]
        # p1=resamp(x[:,:,1::2,:],2)
        # p0, p1 = new_resamp([x[:,:,::2,:], x[:,:,1::2,:]], 2)
        p0, p1 = new_resamp([x[..., ::2], x[..., 1::2]], 2)

    else:
        raise ValueError('Invalid argument type')

    return p0, p1


def resamp(x, type_, shift=1, extmod='per'):
    if type_ in [1, 2]:
        y = resampm(x, type_, shift)

    elif type_ in [3, 4]:
        y = torch.transpose(x, 2, 3)
        y = resampm(y, type_ - 2, shift)
        y = torch.transpose(y, 2, 3)

    else:
        raise ValueError('The second input (type) must be one of {1, 2, 3, 4}')

    return y


def resampm(x, type_, shift=1):
    tic = time()
    N,c,m,n=x.shape

    x = torch.reshape(x, [-1, c, m*n])

    # z = np.zeros((m,n), dtype=np.int64)
    z = torch.zeros((m, n), dtype=torch.int64, device=x.device)
    for j in range(n):
        if type_ == 1:
            k= (shift * j) % m

        else:
            k= (-shift * j) % m

        if k < 0:
            k=k + m
        t1 = torch.arange(k, m, device=x.device)
        t2 = torch.arange(k, device=x.device)
        z[:, j] = torch.cat([t1, t2]) * n + j
        # t1 = np.arange(k, m)
        # t2 = np.arange(k)
        # z[:,j] = np.concatenate([t1, t2]) * n + j

    # z = z.reshape(-1)
    # z = torch.from_numpy(z)
    z = z.view(-1)
    z = z.to(x.device)

    # y = tf.gather(x, z.astype(int), axis=1)
    # y = tf.reshape(y, [-1, m, n, c])
    z = z.reshape((1,1,-1))
    y = torch.gather(x, 2, z.expand(N,c,-1))
    y = torch.reshape(y, [-1, c, m, n])

    # print('This resamp takes:', toc-tic, 'sec. Current time cost on resamp:', total)
    return y


def new_resamp(y, type_, shift=1):
    sample = len(y)
    y = torch.stack(y)
    # print(y.get_shape().as_list())
    if type_ in [3, 4]:
        y = torch.transpose(y, 3, 4)

    # m,n,c=y.get_shape().as_list()[-3:]
    N, c, m, n = y.shape[1:]

    # y = tf.reshape(y, [sample, -1, m*n, c])
    y = torch.reshape(y, [sample, -1, c, m * n])

    # z = np.zeros((m, n), dtype=np.int64)
    z = torch.zeros((m, n), dtype=torch.int64, device=y.device)
    for j in range(n):
        if type_ in [1, 3]:
            k = (shift * j) % m

        else:
            k = (-shift * j) % m

        if k < 0:
            k = k + m
        t1 = torch.arange(k, m, device=y.device)
        t2 = torch.arange(k, device=y.device)
        z[:, j] = torch.cat([t1, t2]) * n + j
        # t1 = np.arange(k, m)
        # t2 = np.arange(k)
        # z[:, j] = np.concatenate([t1, t2]) * n + j

    # z = z.reshape(-1)
    # z = torch.from_numpy(z)  # LongTensor int64
    z = z.view(-1)
    z = z.to(y.device)

    z = torch.reshape(z, (1, 1, 1, -1))
    y = torch.gather(y, 3, z.expand(sample, N, c, -1))

    # y = tf.gather(y, z.astype(int), axis=-2)
    # y = Reshape((m,n,c))(y)
    # y = tf.reshape(y, [sample, -1, m, n, c])
    y = torch.reshape(y, [sample, -1, c, m, n])

    if type_ in [3, 4]:
        y = torch.transpose(y, 3, 4)
    y = [y[i] for i in range(sample)]
    return y


def dfb_filter(device):
    # length 12 filter from Phoong, Kim, Vaidyanathan and Ansari
    v = np.array([0.63, -0.193, 0.0972, -0.0526, 0.0272, -0.0144], dtype=np.float32)
    # Symmetric impulse response
    f = np.concatenate((v[::-1],v))
    # Modulate f
    f[::2] = -f[::2]
    f = torch.from_numpy(f)
    f = f.to(device)
    return f


def new_fbdec_layer(x, f_, type1, type2, extmod='per'):
    sample = len(x)
    # x = tf.concat(x, axis=-1)
    x = torch.cat(x, dim=1)
    ch = x.shape[1] // sample

    # Polyphase decomposition of the input image
    if type1 == 'q':
        # Quincunx polyphase decomposition
        p0, p1 = qpdec_layer(x, type2)

    elif type1 == 'p':
        # Parallelogram polyphase decomposition
        p0, p1 = ppdec_layer(x, type2)

    # Ladder network structure
    y0 = 1 / (2 ** 0.5) * (p0 - sefilter2_layer(p1, f_, f_, extmod, [1, 1]))
    y1 = (-2 ** 0.5) * p1 - sefilter2_layer(y0, f_, f_, extmod)

    # return [y0, y1]
    return [y0[:, i * ch:(i + 1) * ch] for i in range(sample)], [y1[:, i * ch:(i + 1) * ch] for i in range(sample)]


def test():
    device = "cuda"
    img = torch.randn([2000, 1, 64, 64]).to(device)
    nlevs = 2
    ct = ContourDec(1, nlevs, device).to(device)
    t1 = time()
    pri = ct(img)
    pri = ct(img)
    pri = ct(img)
    pri = ct(img)
    pri = ct(img)
    pri = ct(img)
    t2 = time()
    print("total time", t2 - t1)


if __name__ == '__main__':
    test()