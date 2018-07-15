import datetime
import time
from copy import copy
from os import makedirs, remove, listdir
from os.path import exists, join
from shutil import move

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

from helper.dataset import EMNIST_Letters
from helper.model import ModelGenerator


class Helper:
    models_dir = 'saved'
    bests_tmp_dir = 'bests_tmp'
    results = 'results.npy'

    def __init__(self, vae_only=True, hidden_values=[], reg_values=[], drp_values=[], epochs=100):
        # experiments
        if vae_only:
            self.model_types = [True]
        else:
            self.model_types = [True, False]
        self.hidden_values = copy(hidden_values)
        self.reg_values = copy(reg_values)
        self.drp_values = copy(drp_values)
        self.epochs = epochs
        _m = len(self.model_types)
        _h = len(self.hidden_values)
        _r = len(self.reg_values)
        _d = len(self.drp_values)
        self._tot_exp = _m * (_h + _h * _r + _h * _r * _d)

        # dataset
        self.dataset = EMNIST_Letters()
        self.Trn, self.Val, self.Tst = self.dataset.get_novelty_dataset()
        self.Trn_an_idx, self.Val_an_idx, self.Tst_an_idx = self.dataset.get_novelty_anomalies_idx()
        self.Trn_an_idx = np.append(self.Trn_an_idx, self.Val_an_idx)
        self.Trn_idx, self.Val_idx, self.Tst_idx = self.dataset.get_novelty_dataset_labels()
        self.Trn_idx = np.append(self.Trn_idx, self.Val_idx)
        self.Trn_an = np.append(self.Trn, self.Val, axis=0)
        self.Tst_an = copy(self.Tst)
        _, self.Trn_lbls = self.dataset.get_train()
        _, self.Tst_lbls = self.dataset.get_test()

        # save folders
        if not exists(self.models_dir):
            makedirs(self.models_dir)
        if not exists(self.bests_tmp_dir):
            makedirs(self.bests_tmp_dir)

    def train_models(self, verbose=True):
        _start = time.time()
        cur = 0

        if verbose:
            print('\n TRAIN \n')
            print('start at {}'.format(datetime.datetime.fromtimestamp(_start).strftime('%H:%M:%S')))
        for _vae in self.model_types:
            for _hidden in self.hidden_values:
                self._train_model(vae=_vae, hidden=_hidden, reg_val=None, drp_val=None)
                cur += 1
                if verbose:
                    print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
                for _reg in self.reg_values:
                    self._train_model(vae=_vae, hidden=_hidden, reg_val=_reg, drp_val=None)
                    cur += 1
                    if verbose:
                        print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
                    for _drp in self.drp_values:
                        self._train_model(vae=_vae, hidden=_hidden, reg_val=_reg, drp_val=_drp)
                        cur += 1
                        if verbose:
                            print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
        if verbose:
            print('Tot RunTime {}'.format(datetime.datetime.fromtimestamp(time.time() - _start).strftime('%H:%M:%S')))

    def _train_model(self, vae, hidden, reg_val=None, drp_val=None):
        keras.backend.clear_session()

        m_generator = ModelGenerator(vae=vae, hidden=hidden, reg_val=reg_val, drp_val=drp_val)
        name = m_generator.get_name()

        if not exists(join(self.models_dir, '{}_BestW.hdf5'.format(name))):
            checkpointer = ModelCheckpoint(
                filepath=join(self.bests_tmp_dir, name + '_Wep{epoch:03d}_loss{val_loss:.5f}.hdf5'),
                verbose=0,
                save_best_only=True)

            model, history = m_generator.train(epochs=self.epochs, train=self.Trn, validation=self.Val,
                                               callbacks=[checkpointer])

            save_name = join(self.models_dir, name)
            best = sorted(list(filter(lambda w: w.startswith(name) and w.endswith('.hdf5'),
                                      listdir(self.bests_tmp_dir))))[-1]
            move(join(self.bests_tmp_dir, best), '{}_BestW.hdf5'.format(save_name))
            np.save('{}.npy'.format(save_name), history)
            for file in filter(lambda w: w.startswith(name) and w.endswith('.hdf5'), listdir('bests_tmp')):
                remove(join(self.bests_tmp_dir, file))

    def make_tests(self, verbose=True):
        _start = time.time()
        cur = 0
        results = []
        if verbose:
            print('\n TEST \n')
            print('Start at {}'.format(datetime.datetime.fromtimestamp(_start).strftime('%H:%M:%S')))
        for _vae in self.model_types:
            for _hidden in self.hidden_values:
                results.append(self._test(vae=_vae, hidden=_hidden, reg_val=None, drp_val=None))
                cur += 1
                if verbose:
                    print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
                for _reg in self.reg_values:
                    results.append(self._test(vae=_vae, hidden=_hidden, reg_val=_reg, drp_val=None))
                    cur += 1
                    if verbose:
                        print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
                    for _drp in self.drp_values:
                        results.append(self._test(vae=_vae, hidden=_hidden, reg_val=_reg, drp_val=_drp))
                        cur += 1
                        if verbose:
                            print('cur {:3d} of {:2d}'.format(cur, self._tot_exp))
        if verbose:
            print('Tot RunTime {}'.format(datetime.datetime.fromtimestamp(time.time() - _start).strftime('%H:%M:%S')))
        if exists(self.results):
            remove(self.results)
        np.save(self.results, np.array(results))

    def _test(self, vae, hidden, reg_val=None, drp_val=None):
        keras.backend.clear_session()

        self.Trn_an_ts = np.append(self.Trn, self.Val).reshape((len(self.Trn) + len(self.Val), 28 * 28))
        self.Trn_an_ts = tf.convert_to_tensor(self.Trn_an_ts, np.float32)
        self.Tst_an_ts = copy(self.Tst).reshape((len(self.Tst), 28 * 28))
        self.Tst_an_ts = tf.convert_to_tensor(self.Tst_an_ts, np.float32)

        m_generator = ModelGenerator(vae=vae, hidden=hidden, reg_val=reg_val, drp_val=drp_val)
        name = m_generator.get_name()
        m_generator.load_best_w(self.models_dir)
        model = m_generator.get_model()

        loss = model.evaluate(self.Tst, self.Tst, verbose=0)

        trn_pred = model.predict(self.Trn_an).reshape((len(self.Trn_an), 28 * 28))
        trn_pred = tf.convert_to_tensor(trn_pred, np.float32)

        trn_mse = K.eval(mean_squared_error(self.Trn_an_ts, trn_pred))
        th = trn_mse[np.argsort(trn_mse)[-len(self.Trn_an_idx)]]

        tst_pred = model.predict(self.Tst_an).reshape((len(self.Tst_an), 28 * 28))
        tst_pred = tf.convert_to_tensor(tst_pred, np.float32)

        tst_mse = K.eval(mean_squared_error(self.Tst_an_ts, tst_pred))

        [prc, _], [rcl, _], [f1, _], _ = precision_recall_fscore_support(self.Tst_lbls[self.Tst_idx] > 0, tst_mse > th)

        return [name, loss, th, prc, rcl, f1]

    def get_best_reconstruction(self, limit=10):
        if not exists(self.results):
            self.make_tests()
        res = np.load(self.results)
        print('\nBest Reconstruction')
        print('{:26s} |  {:6s}  |  {:6s}  |  {:6s}  |  {:6s}  |  '.format('Name', 'loss', 'prc', 'rcll', 'f1'))
        for i in np.argsort(res[:, 1])[:limit]:
            print('{:26s} |  {:.2f}  |  {:.4f}  |  {:.4f}  |  {:.4f}  |  '.format(res[i][0], float(res[i][1]),
                                                                                  float(res[i][3]), float(res[i][4]),
                                                                                  float(res[i][5])))

    def get_best_precision(self, limit=10):
        if not exists(self.results):
            self.make_tests()
        res = np.load(self.results)
        print('\nBest Precision')
        print('{:26s} |  {:6s}  |  {:6s}  |  {:6s}  |  {:6s}  |  '.format('Name', 'loss', 'prc', 'rcll', 'f1'))
        for i in np.argsort(res[:, -3])[-limit:][::-1]:
            print('{:26s} |  {:.2f}  |  {:.4f}  |  {:.4f}  |  {:.4f}  |  '.format(res[i][0], float(res[i][1]),
                                                                                  float(res[i][3]), float(res[i][4]),
                                                                                  float(res[i][5])))

    def get_best_recall(self, limit=10):
        if not exists(self.results):
            self.make_tests()
        res = np.load(self.results)
        print('\nBest Recall')
        print('{:26s} |  {:6s}  |  {:6s}  |  {:6s}  |  {:6s}  |  '.format('Name', 'loss', 'prc', 'rcll', 'f1'))
        for i in np.argsort(res[:, -2])[-limit:][::-1]:
            print('{:26s} |  {:.2f}  |  {:.4f}  |  {:.4f}  |  {:.4f}  |  '.format(res[i][0], float(res[i][1]),
                                                                                  float(res[i][3]), float(res[i][4]),
                                                                                  float(res[i][5])))

    def get_best_f1(self, limit=10):
        if not exists(self.results):
            self.make_tests()
        res = np.load(self.results)
        print('\nBest f1')
        print('{:26s} |  {:6s}  |  {:6s}  |  {:6s}  |  {:6s}  |  '.format('Name', 'loss', 'prc', 'rcll', 'f1'))
        for i in np.argsort(res[:, -1])[-limit:][::-1]:
            print('{:26s} |  {:.2f}  |  {:.4f}  |  {:.4f}  |  {:.4f}  |  '.format(res[i][0], float(res[i][1]),
                                                                                  float(res[i][3]), float(res[i][4]),
                                                                                  float(res[i][5])))

    def get_best_svmoc(self):
        Trn_svm = np.reshape(self.Trn_an, (len(self.Trn_an), 28 * 28))
        Tst_svm = np.reshape(self.Tst_an, (len(self.Tst_an), 28 * 28))
        print('\nBest SVM-OneClass')
        print('{:10s} |  {:6s}  |  {:6s}  |  {:6s}  |  '.format('gamma', 'Prc', 'rcll', 'f1'))
        for _gamma in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]:
            clf = svm.OneClassSVM(nu=(25 / 825), kernel="rbf", gamma=_gamma)
            clf.fit(Trn_svm)
            pred = clf.predict(Tst_svm) > 0
            [prc, _], [rcl, _], [f1, _], _ = precision_recall_fscore_support(self.Tst_lbls[self.Tst_idx] > 0, pred)
            print('{:10s} |  {:.4f}  |  {:.4f}  |  {:.4f}  | '.format(str(_gamma), prc, rcl, f1))

    def show_reconstrunction(self, vae=True, hidden=2, reg_val=None, drp_val=None, imgs=5):
        m_generator = ModelGenerator(vae=vae, hidden=hidden, reg_val=reg_val, drp_val=drp_val)
        m_generator.load_best_w(self.models_dir)
        model = m_generator.get_model()

        decoded_imgs = model.predict(self.Tst[:imgs], verbose=0)

        plt.figure(figsize=(imgs, 2.5))
        for i in range(imgs):
            ax = plt.subplot(2, imgs, i + 1)
            if i == 2:
                ax.title.set_text('Original')
            plt.imshow(self.Tst[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, imgs, i + 1 + imgs)
            if i == 2:
                ax.title.set_text('Recostruction')
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def get_th(self, vae=True, hidden=2, reg_val=None, drp_val=None):
        if not exists(self.results):
            self.make_tests()
        res = np.load(self.results)
        name = ModelGenerator(vae=vae, hidden=hidden, reg_val=reg_val, drp_val=drp_val).get_name()
        return res[list(res[:, 0]).index(name)][2]

