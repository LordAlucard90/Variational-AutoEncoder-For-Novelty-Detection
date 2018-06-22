from os.path import exists, join
from os import system, makedirs
import struct
import numpy as np


class EMNIST_Letters:
    letters = 26
    float = 'float32'
    dst_path = 'emnist-letters'
    download_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    pth_indices = 'indices.npy'
    x_Trn = x_Val = x_Tst = None
    x_Trn_index = x_Val_index = x_Tst_index = None
    novelty = False
    trn_anomaly_idx = val_anomaly_idx = tst_anomaly_idx = None

    def __init__(self):
        self.pth_tst_img = join(self.dst_path, 'test-images-ubyte')
        self.pth_tst_lbl = join(self.dst_path, 'test-labels-ubyte')
        self.pth_trn_img = join(self.dst_path, 'train-images-ubyte')
        self.pth_trn_lbl = join(self.dst_path, 'train-labels-ubyte')
        if not exists(self.dst_path):
            makedirs(self.dst_path)
        self._get_data()
        self._x_train, self._y_train = self._get_train()
        self._x_test, self._y_test = self._get_test()

    def _get_data(self):
        if not exists('gzip.zip'):
            system('wget --no-check-certificate {}'.format(self.download_url))
        if not exists(self.pth_tst_img):
            system('unzip -p gzip.zip gzip/emnist-letters-test-images-idx3-ubyte.gz > {}/test-images-ubyte.gz'
                   .format(self.dst_path))
            system('gzip -d {}/test-images-ubyte.gz'.format(self.dst_path))
        if not exists(self.pth_tst_lbl):
            system('unzip -p gzip.zip gzip/emnist-letters-test-labels-idx1-ubyte.gz  > {}/test-labels-ubyte.gz'
                   .format(self.dst_path))
            system('gzip -d {}/test-labels-ubyte.gz'.format(self.dst_path))
        if not exists(self.pth_trn_img):
            system('unzip -p gzip.zip gzip/emnist-letters-train-images-idx3-ubyte.gz > {}/train-images-ubyte.gz'
                   .format(self.dst_path))
            system('gzip -d {}/train-images-ubyte.gz'.format(self.dst_path))
        if not exists(self.pth_trn_lbl):
            system('unzip -p gzip.zip gzip/emnist-letters-train-labels-idx1-ubyte.gz > {}/train-labels-ubyte.gz'
                   .format(self.dst_path))
            system('gzip -d {}/train-labels-ubyte.gz'.format(self.dst_path))

    def _get_train(self):
        images, labels = self._load_data(self.pth_trn_img, self.pth_trn_lbl)
        return self._regularize(images), labels

    def _get_test(self):
        images, labels = self._load_data(self.pth_tst_img, self.pth_tst_lbl)
        return self._regularize(images), labels

    @staticmethod
    def _load_data(pth_images, pth_labels):
        with open(pth_labels, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
        with open(pth_images, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
        return np.transpose(img, (0, 2, 1)), lbl - 1

    @staticmethod
    def _regularize(images):
        return np.reshape(images, (len(images), 28, 28, 1)).astype('float32') / 255

    def get_train(self):
        return self._x_train, self._y_train

    def get_test(self):
        return self._x_test, self._y_test

    def get_novelty_dataset(self):
        self._generate_novelty_dataset()
        return self.x_Trn, self.x_Val, self.x_Tst

    def get_novelty_dataset_labels(self):
        self._generate_novelty_dataset()
        return self.x_Trn_index, self.x_Val_index, self.x_Tst_index

    def get_novelty_anomalies_idx(self):
        self._generate_novelty_dataset()
        return self.trn_anomaly_idx, self.val_anomaly_idx, self.tst_anomaly_idx

    def _generate_novelty_dataset(self):
        if not self.novelty:
            if not exists(self.pth_indices):
                self.x_vals_Tr = [[] for i in range(self.letters)]
                self.x_vals_Ts = [[] for i in range(self.letters)]
                self.x_Trn_index = []
                self.x_Val_index = []
                self.x_Tst_index = []

                for i in range(len(self._x_train)):
                    self.x_vals_Tr[self._y_train[i]].append(i)
                for i in range(len(self._x_test)):
                    self.x_vals_Ts[self._y_test[i]].append(i)

                for l in range(self.letters):
                    np.random.shuffle(self.x_vals_Tr[l])
                    np.random.shuffle(self.x_vals_Ts[l])
                    if l > 0:
                        self.x_Trn_index.extend(self.x_vals_Tr[l][:5]) # ~ 3%
                        self.x_Val_index.append(self.x_vals_Tr[l][6])  # ~ 3%
                        self.x_Tst_index.append(self.x_vals_Ts[l][0])  # ~ 3%
                    else:
                        self.x_Trn_index.extend(self.x_vals_Tr[l][:4000])
                        self.x_Val_index.extend(self.x_vals_Tr[l][4000:])
                        self.x_Tst_index.extend(self.x_vals_Ts[l])

                self.trn_anomaly_idx = self.x_Trn_index[-(5 * (self.letters - 1)):]
                self.val_anomaly_idx = self.x_Val_index[-(self.letters - 1):]
                self.tst_anomaly_idx = self.x_Tst_index[-(self.letters - 1):]


                _index = {'trn': self.x_Trn_index,
                          'val': self.x_Val_index,
                          'tst': self.x_Tst_index,
                          'val_tr': self.x_vals_Tr,
                          'val_ts': self.x_vals_Ts,
                          'anomaly': {'trn': self.trn_anomaly_idx,
                                      'val': self.val_anomaly_idx,
                                      'tst': self.tst_anomaly_idx}
                          }
                np.save(self.pth_indices, _index)

            else:
                _index = np.load(self.pth_indices)[()]
                self.x_vals_Tr = _index['val_tr']
                self.x_vals_Ts = _index['val_ts']
                self.x_Trn_index = _index['trn']
                self.x_Val_index = _index['val']
                self.x_Tst_index = _index['tst']
                self.trn_anomaly_idx = _index['anomaly']['trn']
                self.val_anomaly_idx = _index['anomaly']['val']
                self.tst_anomaly_idx = _index['anomaly']['tst']

            self.x_Trn_index = np.array(self.x_Trn_index)
            self.x_Val_index = np.array(self.x_Val_index)
            self.x_Tst_index = np.array(self.x_Tst_index)
            self.trn_anomaly_idx = np.array(self.trn_anomaly_idx)
            self.val_anomaly_idx = np.array(self.val_anomaly_idx)
            self.tst_anomaly_idx = np.array(self.tst_anomaly_idx)

            np.random.shuffle(self.x_Trn_index)
            np.random.shuffle(self.x_Val_index)
            np.random.shuffle(self.x_Tst_index)

            self.x_Trn = self._x_train[self.x_Trn_index]
            self.x_Val = self._x_train[self.x_Val_index]
            self.x_Tst = self._x_test[self.x_Tst_index]

            self.novelty = True

