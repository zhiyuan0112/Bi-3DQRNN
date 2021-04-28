from .qrnn import QRNNREDC3D
from .qrnn import FQRNNREDC3D
from .qrnn import BiFQRNNREDC3D
from .qrnn import IQRNNREDC3D

"""Define commonly used architecture"""
def qrnn_16_5():
    net = QRNNREDC3D(1, 16, 5, [1,3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def fqrnn_16_5():
    net = FQRNNREDC3D(1, 16, 5, [1,3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def iqrnn_16_5():
    net = IQRNNREDC3D(1, 16, 5, [1,3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def biqrnn_16_5():
    net = BiFQRNNREDC3D(1, 16, 5, [1,3])
    net.use_2dconv = False
    net.bandwise = False
    return net
