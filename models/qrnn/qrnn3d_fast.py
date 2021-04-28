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


"""F pooling"""
class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F.sigmoid()

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []
        
        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
                ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)
        
        # return concatenated hidden states
        return torch.cat(h_time, dim=2)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F1.sigmoid(), F2.sigmoid()

    def forward(self, inputs, fname=None):        
        h = None 
        Z, F1, F2 = self._conv_step(inputs)
        hsl = [] ; hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep            
            h = self._rnn_step(z, f, h)
            hsl.append(h)
        
        h = None
        for time, (z, f) in enumerate((zip(
            reversed(zs), reversed(F2.split(1, 2))
            ))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)
        
        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {'z':Z, 'fl':F1, 'fr':F2, 'hsl':hsl, 'hsr':hsr}
            torch.save(stats_dict, fname)
        return hsl + hsr


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, Conv3d(in_channels, hidden_channels*3, k, s, p))


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, DeConv3d(in_channels, hidden_channels*3, k, s, p, bias=bias))


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, Conv3d(in_channels, hidden_channels*2, k, s, p))


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, DeConv3d(in_channels, hidden_channels*2, k, s, p))


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2)):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BNUpsampleConv3d(in_channels, hidden_channels*2, k, s, p, upsample))


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
    # data = Variable(torch.randn(12,1,197,32,32)).cuda()
    # data = Variable(torch.randn(1,1,31,512,512), volatile=True).cuda()
    # data = Variable(torch.randn(1,1,197,216,216), volatile=True).cuda()
    net = QRNNREDC3D(1, 16, 5, [1,3])
    net.cuda()
    print(net)
    out = net(data)
    # nn.MSELoss()(out, Variable(torch.ones_like(out.data), volatile=True)).backward()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     out = net(data)
    # with open('prof.txt', 'w') as f:
    #     f.write(str(prof))
    
    import ipdb; ipdb.set_trace()
