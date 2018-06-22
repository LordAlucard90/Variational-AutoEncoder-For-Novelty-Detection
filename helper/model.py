from os.path import join

from keras.layers import Input, Lambda, Flatten, Dense, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras import backend as K


class ModelGenerator:
    def __init__(self, vae=True, hidden=2, reg_val=None, drp_val=None):
        self.vae = vae
        self.hidden = hidden
        self.reg_val = reg_val
        self.drp_val = drp_val
        self._generate_name()
        self._generate_model()

    def get_name(self):
        return self.name

    def load_w(self, search_path):
        self.model.load_weights(join(search_path, '{}.hdf5'.format(self.name)))

    def load_best_w(self, search_path):
        self.model.load_weights(join(search_path, '{}_BestW.hdf5'.format(self.name)))

    def get_model(self):
        return self.model

    def train(self, epochs, train, validation, callbacks):
        print("Training.. ", end='')
        history = self.model.fit(train, train,
                                 epochs=epochs,
                                 batch_size=128,
                                 shuffle=True,
                                 validation_data=(validation, validation),
                                 callbacks=callbacks,
                                 verbose=0)
        print("Done")
        return self.model, history.history

    def _generate_name(self):
        if self.vae:
            self.name = 'vae_H{:02d}'.format(self.hidden)  # Variational AutoEncoder
        else:
            self.name = 'sae_H{:02d}'.format(self.hidden)  # Standard AutoEncoder
        if self.reg_val is not None:
            self.name = '{}_reg{}'.format(self.name, self.reg_val)
            if self.drp_val is not None:
                self.name = '{}_drp{}'.format(self.name, self.drp_val)

    def _generate_model(self):
        if self.reg_val is not None:
            if self.drp_val is not None:
                self._get_regularized_and_dropout()
            else:
                self._get_regularized()
        else:
            self._get_standard()

    @staticmethod
    def _sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.0)
        return mean + K.exp(log_var) * epsilon

    def _get_standard(self):
        input_img = Input(shape=(28, 28, 1))
        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)

        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)

        encoder_out = Flatten()(Concatenate()([encoder_branch_left, encoder_branch_right]))
        encoder_out = Dense(128, activation='relu')(encoder_out)

        if self.vae:
            mean = Dense(self.hidden, name='mean')(encoder_out)
            log_var = Dense(self.hidden, name='log_var')(encoder_out)
            mirror = Lambda(self._sampling)([mean, log_var])
        else:
            mirror = Dense(self.hidden, name='log_var')(encoder_out)

        decoder = Dense(128, activation='relu')(mirror)
        decoder = Dense(16 * 4 * 4, activation='relu')(decoder)
        decoder = Reshape((4, 4, 16))(decoder)

        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu')(decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_branch_left)

        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu')(decoder)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu')(decoder_branch_right)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(decoder_branch_right)

        out = Concatenate()([decoder_branch_left, decoder_branch_right])
        out = Conv2D(16, (3, 3), activation='relu', padding='same')(out)
        out_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(out)

        self.model = Model(input_img, out_img)

        if self.vae:
            def my_loss(y_true, y_pred):
                xent = 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
                kl = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
                return K.mean(xent + kl)
        else:
            def my_loss(y_true, y_pred):
                return 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

        self.model.compile(optimizer='rmsprop', loss=my_loss)

    def _get_regularized(self):
        input_img = Input(shape=(28, 28, 1))
        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)

        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same',
                                      kernel_regularizer=l2(self.reg_val))(
            encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same',
                                      kernel_regularizer=l2(self.reg_val))(
            encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)

        encoder_out = Flatten()(Concatenate()([encoder_branch_left, encoder_branch_right]))
        encoder_out = Dense(128, activation='relu', kernel_regularizer=l2(self.reg_val))(encoder_out)

        if self.vae:
            mean = Dense(self.hidden)(encoder_out)
            log_var = Dense(self.hidden)(encoder_out)
            mirror = Lambda(self._sampling)([mean, log_var])
        else:
            mirror = Dense(self.hidden)(encoder_out)

        decoder = Dense(128, activation='relu', kernel_regularizer=l2(self.reg_val))(mirror)
        decoder = Dense(16 * 4 * 4, activation='relu', kernel_regularizer=l2(self.reg_val))(decoder)
        decoder = Reshape((4, 4, 16))(decoder)

        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(decoder)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)

        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder_branch_right)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', padding='same',
                                               kernel_regularizer=l2(self.reg_val))(decoder_branch_right)  # -> 28

        out = Concatenate()([decoder_branch_left, decoder_branch_right])
        out = Conv2D(16, (3, 3), activation='relu', padding='same')(out)
        out_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(out)

        self.model = Model(input_img, out_img)

        if self.vae:
            def my_loss(y_true, y_pred):
                xent = 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
                kl = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
                return K.mean(xent + kl)
        else:
            def my_loss(y_true, y_pred):
                return 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

        self.model.compile(optimizer='rmsprop', loss=my_loss)

    def _get_regularized_and_dropout(self):
        input_img = Input(shape=(28, 28, 1))
        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            encoder_branch_left)
        encoder_branch_left = Dropout(self.drp_val)(encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            encoder_branch_left)
        encoder_branch_left = Dropout(self.drp_val)(encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)

        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same',
                                      kernel_regularizer=l2(self.reg_val))(
            encoder_branch_right)
        encoder_branch_right = Dropout(self.drp_val)(encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same',
                                      kernel_regularizer=l2(self.reg_val))(
            encoder_branch_right)
        encoder_branch_right = Dropout(self.drp_val)(encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder_branch_right)

        encoder_out = Flatten()(Concatenate()([encoder_branch_left, encoder_branch_right]))
        encoder_out = Dense(128, activation='relu', kernel_regularizer=l2(self.reg_val))(encoder_out)

        if self.vae:
            mean = Dense(self.hidden)(encoder_out)
            log_var = Dense(self.hidden)(encoder_out)
            mirror = Lambda(self._sampling)([mean, log_var])
        else:
            mirror = Dense(self.hidden)(encoder_out)

        decoder = Dense(128, activation='relu', kernel_regularizer=l2(self.reg_val))(mirror)
        decoder = Dense(16 * 4 * 4, activation='relu', kernel_regularizer=l2(self.reg_val))(decoder)
        decoder = Reshape((4, 4, 16))(decoder)
        decoder = Dropout(self.drp_val)(decoder)

        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(decoder)
        decoder_branch_left = Dropout(self.drp_val)(decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)
        decoder_branch_left = Dropout(self.drp_val)(decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=l2(self.reg_val))(
            decoder_branch_left)

        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder)
        decoder_branch_right = Dropout(self.drp_val)(decoder_branch_right)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', kernel_regularizer=l2(self.reg_val))(
            decoder_branch_right)
        decoder_branch_right = Dropout(self.drp_val)(decoder_branch_right)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu', padding='same',
                                               kernel_regularizer=l2(self.reg_val))(decoder_branch_right)

        out = Concatenate()([decoder_branch_left, decoder_branch_right])
        out = Conv2D(16, (3, 3), activation='relu', padding='same')(out)
        out_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(out)

        self.model = Model(input_img, out_img)

        if self.vae:
            def my_loss(y_true, y_pred):
                xent = 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
                kl = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
                return K.mean(xent + kl)
        else:
            def my_loss(y_true, y_pred):
                return 28 * 28 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

        self.model.compile(optimizer='rmsprop', loss=my_loss)
