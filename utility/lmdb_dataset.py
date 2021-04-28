import sys
import os
import os.path

import torch.utils.data as data
import numpy as np
import caffe
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # cache_file = '_cache_' + db_path.replace('/', '_')
        # if os.path.isfile(cache_file):
        #     self.keys = pickle.load(open(cache_file, "rb"))
        # else:
        #     with self.env.begin(write=False) as txn:
        #         self.keys = [key for key, _ in txn.cursor()]
        #     pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)

        return x

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


if __name__ == '__main__':
    pass
    