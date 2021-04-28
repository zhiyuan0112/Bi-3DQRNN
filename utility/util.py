# utility functions for generating hdf5 datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# # https://github.com/pytorch/pytorch/issues/3415
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

import h5py
import os
import random

from itertools import product
from scipy.io import loadmat
from functools import partial
from scipy.ndimage import zoom
from matplotlib.widgets import Slider
from PIL import Image


def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)    

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
    
    V = np.zeros([int(TotalPatNum)]+ksizes); # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = data[s2].reshape(-1)
        
    return V


def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


def rand_crop(img, cropx, cropy):
    _,y,x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1:y1+cropy, x1:x1+cropx]


def sequetial_process(*fns):
    """
    Integerate all process functions
    """
    def processor(data):
        for f in fns:
            data = f(data)
        return data
    return processor


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def frame_diff(frames):
    diff_frames = frames[1:, ...] - frames[:-1, ...]
    return diff_frames


def visualize(filename, matkey, load=loadmat, preprocess=None):
    """
    Visualize a preprecessed hyperspectral image
    """
    if not preprocess:
        preprocess = lambda identity: identity
    mat = load(filename)
    data = preprocess(mat[matkey])
    print(data.shape)
    print(np.max(data), np.min(data))

    data = np.squeeze(data[:,:,:])
    Visualize3D(data)
    # Visualize3D(np.squeeze(data[:,0,:,:]))

def Visualize3D(data, meta=None):
    data = np.squeeze(data)

    for ch in range(data.shape[0]):        
        data[ch, ...] = minmax_normalize(data[ch, ...])
    
    print(np.max(data), np.min(data))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    # l = plt.imshow(data[frame,:,:])
    
    l = plt.imshow(data[frame,:,:], cmap='gray') #shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, data.shape[0]-1, valinit=0)

    def update(val):
        frame = int(np.around(sframe.val))
        l.set_data(data[frame,:,:])
        if meta is not None:
            axframe.set_title(meta[frame])

    sframe.on_changed(update)

    plt.show()


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def cal_img_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ( (im1.astype(np.float) - im2.astype(np.float)) ** 2 ).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def bwmse_loss(x, y):
    """Bandwise Mean Square Error Loss
    Note:
        This module not returns Variable
    """
    C = x.shape[-3]
    bwmse = []
    for ch in range(C):
        bwmse.append(torch.mean((x[...,ch,:,:] - y[...,ch,:,:])**2).item())
    return bwmse


def save_images(ground_truth, noisy_image, clean_image, filepath):
    # assert the pixel value range is 0-255
    _, im_h, im_w = noisy_image.shape
    ground_truth = ground_truth.reshape((im_h, im_w))
    noisy_image = noisy_image.reshape((im_h, im_w))
    clean_image = clean_image.reshape((im_h, im_w))
    cat_image = np.column_stack((noisy_image, clean_image))
    cat_image = np.column_stack((ground_truth, cat_image))
    cat_image = np.clip(255 * cat_image, 0, 255).astype('uint8')
    im = Image.fromarray(cat_image).convert('L')
    im.save(filepath, 'png')


def save_image(im,filepath):
    _, im_h, im_w, _ = im.shape
    im = im.reshape(im_h,im_w)
    img = Image.fromarray(im.astype('uint8')).convert('L')
    img.save(filepath, 'png')


def data_augmentation(image, mode=None):
    """
    Warning: this function is not available for pytorch DataLoader now,
    since it only return a view of original array 
    which is currently not supported by DataLoader.

    To use data augmentation in data with type of numpy.ndarray, 
    you need first transform the numpy array into PIL.Image, then 
    use torchvision.transforms to augment data.

    Data augmentation in numpy level rather than PIL.Image level
    """
    axes = (-2, -1)
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = np.flipud(image)
    
    return image


import threading
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


if __name__ == '__main__':
    pass
