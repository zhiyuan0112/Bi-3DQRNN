"""Highway"""
import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np

from functools import partial

if __name__ == '__main__':    
    from combinations import *
    from utils import *
else:
    from .combinations import *
    from .utils import *


"""IFO pooling"""
class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer

    def _conv_step(self, X):
        gates = self.conv(X)
        # Z, F, O: [batch_size x hidden_channels x D x H x W]
        Z, F, O, I = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F.sigmoid(), O.sigmoid(), I.sigmoid()

    def _rnn_step(self, z, f, o, c, i):
        # uses 'fo pooling' at each time step
        c_ = i * z if c is None else f * c + i * z
        return c_, (o * c_) 	# return c_t and h_t

    def forward(self, X, reverse=False):
        c = None
        Z, F, O, I = self._conv_step(X)
        c_time, h_time = [], []
        zs = Z.split(1, 2)
        
        if not reverse:
            for time, (z, f, o, i) in enumerate(zip(zs, F.split(1, 2), O.split(1, 2), I.split(1, 2)
                )):  # split along timestep            
                c, h = self._rnn_step(z, f, o, c, i)
                h_time.append(h)
        else:
            for time, (z, f, o, i) in enumerate((zip(
                reversed(zs), reversed(F.split(1, 2)), reversed(O.split(1, 2)), reversed(I.split(1, 2))
                ))):  # split along timestep
                c, h = self._rnn_step(z, f, o, c, i)
                h_time.insert(0, h)
        
        # return concatenated cell & hidden states
        return torch.cat(h_time, dim=2)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, X):
        gates = self.conv(X)
        Z, F1, F2, O1, O2, I1, I2 = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F1.sigmoid(), F2.sigmoid(), O1.sigmoid(), O2.sigmoid(), I1.sigmoid(), I2.sigmoid()

    def forward(self, X):        
        h = None; c = None
        Z, F1, F2, O1, O2, I1, I2 = self._conv_step(X)
        hsl = [] ; hsr = []
        zs = Z.split(1, 2)

        for time, (z, f, o, i) in enumerate(zip(zs, F1.split(1, 2), O1.split(1, 2), I1.split(1, 2)
            )):  # split along timestep            
            c, h = self._rnn_step(z, f, o, c, i)
            hsl.append(h)
        
        h = None; c = None
        for time, (z, f, o, i) in enumerate((zip(
            reversed(zs), reversed(F2.split(1, 2)), reversed(O2.split(1, 2)), reversed(I2.split(1, 2))
            ))):  # split along timestep
            c, h = self._rnn_step(z, f, o, c, i)
            hsr.insert(0, h)
        
        # return concatenated hidden states
        return torch.cat(hsl, dim=2) + torch.cat(hsr, dim=2)


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, Conv3d(in_channels, hidden_channels*7, k, s, p))


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, DeConv3d(in_channels, hidden_channels*7, k, s, p, bias=bias))


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, Conv3d(in_channels, hidden_channels*4, k, s, p))


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, DeConv3d(in_channels, hidden_channels*4, k, s, p))


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2)):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BNUpsampleConv3d(in_channels, hidden_channels*4, k, s, p, upsample))


QRNN3DEncoder = partial(
    QRNN3DEncoder, 
    QRNNConv3D=QRNNConv3D)

QRNN3DDecoder = partial(
    QRNN3DDecoder, 
    QRNNDeConv3D=QRNNDeConv3D, 
    QRNNUpsampleConv3d=QRNNUpsampleConv3d)

QRNNREDC3D = partial(
    QRNNREDC3D, 
    BiQRNNConv3D=BiQRNNConv3D, 
    BiQRNNDeConv3D=BiQRNNDeConv3D,
    QRNN3DEncoder=QRNN3DEncoder,
    QRNN3DDecoder=QRNN3DDecoder
)


if __name__ == '__main__':
    from torch.autograd import Variable
    data = Variable(torch.randn(12,1,31,64,64)).cuda()
    # data = Variable(torch.randn(1,1,31,256,256)).cuda()
    net = QRNNREDC3D(1, 16, 5, [1,3])
    net.cuda()
    print(net)
    out = net(data)
    nn.MSELoss()(out, Variable(torch.ones_like(out.data), volatile=True)).backward()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     out = net(data)
    # with open('prof.txt', 'w') as f:
    #     f.write(str(prof))
    
    import ipdb; ipdb.set_trace()
