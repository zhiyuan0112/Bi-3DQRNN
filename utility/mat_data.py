"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize


def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        mat = load(filepath)
        
        data = func(mat[matkey][...])
        savemat(join(newdir, fn), {'gt':data.transpose((2,1,0))})


def create_icvl_sr():
    basedir = '/data1/liangzhiyuan/Data/ICVL'
    datadir = join(basedir, 'Testing')
    newdir = join(basedir, 'icvl_512_gt')
    fnames = os.listdir(datadir)
    
    def func(data):
        data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 512, 512)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'rad', func=func)


if __name__ == '__main__':
    # create_icvl_sr()
    pass
