"""Create lmdb dataset"""
from util import *
import lmdb
import caffe
import scipy.io as sio
import numpy as np


def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2021):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        data = minmax_normalize(data)
        # data = minmax_normalize(data.transpose((2,0,1)))  # Remote sensed data.
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])
        for i in range(len(scales)):
            temp = zoom(data, zoom=(1, scales[i], scales[i]))
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset.
    data = sio.loadmat(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]
    
    print(data.shape)

    # We need to prepare the database for the size. We'll set it 2 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = sio.loadmat(datadir + fn)[matkey] 
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)        
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')


def create_lmdb_test(
    datadir, fns, name, matkey,
    preprocess, 
    load=h5py.File, 
    seed=2021):
    """
    Create Augmented Dataset
    """
    np.random.seed(seed)

    # calculate the shape of dataset
    data = sio.loadmat(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]
    
    print(data.shape)

    # We need to prepare the database for the size. We'll set it 2 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 1.25
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i, fn in enumerate(fns):
            X = preprocess(load(datadir + fn)[matkey])
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(i*N+j)
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')


def create_icvl64_31():
    print('create icvl64_31...')
    datadir = '/data1/liangzhiyuan/Data/ICVL/Training/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/data1/liangzhiyuan/Data/ICVL/Training', 'rad', 
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25, 0.125),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32), (31, 32, 32)],
        load=h5py.File, augment=True,
    )

def create_Pavia():
    print('create Pavia...')
    datadir = '/media/liangzhiyuan/liangzy/qrnn3d/data/pavia/train/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    # fns = ['Pavia.mat']

    create_lmdb_train(
        datadir, fns, '/media/liangzhiyuan/liangzy/qrnn3d/data/pavia/pavia', 'pavia', 
        crop_sizes=None,
        scales=(1,),
        ksizes=(102, 64, 64),
        strides=[(102, 64, 64)],
        load=loadmat, augment=True, 
    )

def create_PaviaU():
    print('create Pavia...')
    datadir = '/data1/liangzhiyuan/Data/remote/train/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/data1/liangzhiyuan/Data/remote/train/paviau', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(101, 64, 64),
        strides=[(101, 64, 64)],
        load=loadmat, augment=True,
    )

def create_Salinas():
    print('create Salinas...')
    datadir = '/data1/liangzhiyuan/Data/remote/train/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/data1/liangzhiyuan/Data/remote/train/salinas', 'hsi', 
        crop_sizes=None,
        scales=(1,),
        ksizes=(197, 64, 64),
        strides=[(197, 32, 32)],
        load=loadmat, augment=True,
    )

def create_Indian():
    print('create Indian...')
    datadir = '/data1/liangzhiyuan/Data/remote/train/'
    fns = os.listdir(datadir)
    fns = ['indian_pines1.mat', 'indian_pines2.mat']

    create_lmdb_train(
        datadir, fns, '/data1/liangzhiyuan/Data/remote/train/indian', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(200, 36, 36),
        strides=[(200, 18, 18)],
        load=loadmat, augment=True,
    )

def create_Urban():
    print('create Urban...')
    datadir = '/data1/liangzhiyuan/Data/remote/train/'
    fns = os.listdir(datadir)
    fns = ['urban1.mat', 'urban2.mat']

    create_lmdb_train(
        datadir, fns, '/data1/liangzhiyuan/Data/remote/train/urban', 'hsi',
        crop_sizes=None,
        scales=(1,),
        ksizes=(162, 64, 64),
        strides=[(162, 32, 32)],
        load=loadmat, augment=True,
    )



if __name__ == '__main__':
    # create_icvl64_31()
    create_Pavia()
    pass
