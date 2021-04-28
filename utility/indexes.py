import numpy as np
import torch
from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from functools import partial


class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
cal_bwssim = Bandwise(structural_similarity)


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)    
    return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam


"""Depreciated"""
def cal_psnr(mse):
    return 10 * np.log10(1 / mse)


def mpsnr(bwmse, verbose=False):
    psnrs = []
    for mse in bwmse:
        cur_psnr = cal_psnr(mse)
        psnrs.append(cur_psnr)
    
    if not verbose:
        return np.mean(psnrs)
    else:
        return np.mean(psnrs), psnrs
