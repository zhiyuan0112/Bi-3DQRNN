from sys import path
import scipy.io as sio

from util import crop_center, minmax_normalize

path = '/BIT/BIT/liangzhiyuan/remote/'
# filenames = os.listdir(path)
filename = 'Pavia.mat'
matkey = 'pavia'
crop_size = (256, 256)

img = sio.loadmat(path + filename)[matkey]
img = minmax_normalize(img.transpose(2,0,1))
print(img.shape)
_, y, x = img.shape
startx = x//2-(crop_size[0]//2)
starty = y//2-(crop_size[1]//2)

# Create test region.
path_test = '/BIT/BIT/liangzhiyuan/pavia/test/pavia_test.mat'
img_test = crop_center(img, crop_size[0], crop_size[1])
sio.savemat(path_test, {'gt':img_test.transpose(1,2,0)})


# Create train regions.
img_train1 = img[:, 0:starty, :]
img_train2 = img[:, :, 0:startx]
img_train3 = img[:, :, startx+crop_size[0]-1:x]
img_train4 = img[:, starty+crop_size[1]-1:y, :]
path_train = '/BIT/BIT/liangzhiyuan/pavia/train/pavia'
sio.savemat(path_train + '1.mat', {'pavia':img_train1})
sio.savemat(path_train + '2.mat', {'pavia':img_train2})
sio.savemat(path_train + '3.mat', {'pavia':img_train3})
sio.savemat(path_train + '4.mat', {'pavia':img_train4})
