# There are functions for creating a train and validation iterator.
import torch
import torchvision
import random

try: 
    from .util import *
except:
    from util import *

from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomHorizontalFlip, Scale
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset, TensorDataset, ResampleDataset

from PIL import Image
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter

# Define Transforms
class RandomFlip(object):
    """flip the given PIL.Image randomly with a probability of 0.5.
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4
    """

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        mode = random.randint(0, 4)
        if random.random() < 0.5:
            return img.transpose(mode)
        return img


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True: 
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


"""For Super-Resolution"""
class SRDegrade(object):
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
    
    def __call__(self, img):        
        img = zoom(img, zoom=(1, 1./self.scale_factor, 1./self.scale_factor))
        img = zoom(img, zoom=(1, self.scale_factor, self.scale_factor))
        return img


class GaussianBlur(object):
    def __init__(self, ksize=8, sigma=3):
        self.sigma = sigma
        self.truncate = (((ksize - 1)/2)-0.5)/sigma

    def __call__(self, img):
        img = gaussian_filter(img, sigma=self.sigma, truncate=self.truncate)
        return img


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            temp = hsi[None]/1.0
            img = torch.from_numpy(temp)        
        return img.float()


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform
    
    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key].transpose((2,0,1)))
            gt = self.transform(mat[self.gt_key].transpose((2,0,1)))
        else:
            input = mat[self.input_key].transpose((2,0,1))
            gt = mat[self.gt_key].transpose((2,0,1))
        # input = torch.from_numpy(input[None]).float()
        input = torch.from_numpy(input).float()
        # gt = torch.from_numpy(gt[None]).float()  # for 3D net
        gt = torch.from_numpy(gt).float()

        return input, gt


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key
    
    def __call__(self, mat):
        item = mat[self.key].transpose((2,0,1))
        return item

# Define Datasets

def load_npy(filepath):
    assert '.npy' in filepath
    if not os.path.exists(filepath):
        print("[!] Data file not exists")
        sys.exit(1)
    
    print("[*] Loading data...")
    data = np.load(filepath)
    # np.random.shuffle(data)
    print("[*] Load successfully...")
    return data


class HDF5Dataset(Dataset):
    """
    Warning: lower version of HDF5 may not support multi-thread 
    which definitely degrades the performance of dataloader.

    Dataset wrapping data from hdf5 dataset object
    Args:
        h5data (HDF5 dataset): HDF5 dataset object
    """
    def __init__(self, h5data):
        self.h5data = h5data
        
    def __getitem__(self, index):
        return self.h5data[index, ...]

    def __len__(self):
        return len(self.h5data)


class DatasetFromFolder(Dataset):
    """Wrap data from image folder"""
    def __init__(self, data_dir, suffix='png'):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir) 
            if fn.endswith(suffix)
        ]

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('L')
        return img

    def __len__(self):
        return len(self.filenames)


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', size=None):
        super(MatDataFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir)
            if fn.endswith(suffix)
        ]
        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        mat = self.load(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)


def get_train_valid_loader(dataset,
                           batch_size,
                           train_transform=None,
                           valid_transform=None,
                           valid_size=None,
                           shuffle=True,
                           verbose=False,
                           num_workers=1,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over any pytorch dataset. A sample 
    of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: full dataset which contains training and validation data
    - batch_size: how many samples per batch to load. (train, val)
    - train_transform/valid_transform: callable function 
      applied to each sample of dataset. default: transforms.ToTensor().
    - valid_size: should be a integer in the range [1, len(dataset)].
    - shuffle: whether to shuffle the train/validation indices.
    - verbose: display the verbose information of dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    
    # define transform
    default_transform = lambda item: item  # identity maping
    train_transform = train_transform or default_transform
    valid_transform = valid_transform or default_transform

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}

    train_dataset = TransformDataset(
        SplitDataset(dataset, partitions, initial_partition='Train'),
        train_transform
    )

    valid_dataset = TransformDataset(
        SplitDataset(dataset, partitions, initial_partition='Valid'),
        valid_transform
    )

    train_loader = DataLoader(train_dataset,
                    batch_size=batch_size[0], shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = DataLoader(valid_dataset, 
                    batch_size=batch_size[1], shuffle=False, 
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)


def get_train_valid_dataset(dataset, valid_size=None):                           
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}

    train_dataset = SplitDataset(dataset, partitions, initial_partition='Train')
    valid_dataset = SplitDataset(dataset, partitions, initial_partition='Valid')

    return (train_dataset, valid_dataset)


class ImageTransformDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):        
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == '__main__':
    pass
