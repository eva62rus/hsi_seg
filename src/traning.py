from __future__ import print_function
import os
import glob
import skimage.io as io
import tensorflow as tf
import tensorflow.keras as Keras
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, ReLU
from tensorflow.keras.layers import Input, concatenate, AveragePooling2D, Conv2D, MaxPooling2D, Conv2DTranspose, \
    BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.optimizers import *
import math
from tensorflow.keras.models import load_model
from segmentation_models.base import Loss
import numpy as np

# print('Введите число эпох обучения:')
EPOCHS = 200  # int(input())
SMOOTH = 1e-5
s = 5  # 100


def conv_block(m, dim, bn, res, do=0, second_conv=True):
    n = Conv2D(dim, 3, activation='linear', padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = ReLU()(n)
    n = Dropout(do)(n) if do else n
    if second_conv:
        n = Conv2D(dim, 3, activation='linear', padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        n = ReLU()(n)
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, do, bn, mp, up, res, second_conv=True):
    if depth > 0:
        n = conv_block(m, dim, bn, res, do, second_conv)
        m = MaxPooling2D()(n) if mp else AveragePooling2D()(n)
        m = level_block(m, int(inc * dim), depth - 1, inc, do, bn, mp, up, res, second_conv)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation='relu', padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation='relu', padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, bn, res, do, second_conv)
    else:
        m = conv_block(m, dim, bn, res, do, second_conv)
    return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2.,
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, second_conv=True):
    i = Keras.Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, dropout, batchnorm, maxpool, upconv, residual, second_conv)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Keras.Model(inputs=i, outputs=o)


max_bright_code = 256


def train_generator(train_path, image_folder, mask_folder, first_image, num_image, batch_size,
                    image_size):
    X = np.zeros((batch_size, image_size, image_size, 4))
    Y = np.zeros((batch_size, image_size, image_size, 1))
    while 1:
        for i in range(first_image, first_image + num_image, batch_size):
            for batch_index in range(batch_size):
                imgNum = (i + batch_index) % (first_image + num_image)
                img = io.imread(os.path.join(train_path + image_folder, "%d.tif" % imgNum))
                img = img / max_bright_code
                mask = io.imread(os.path.join(train_path + mask_folder, "%d.tif" % imgNum))
                if mask.ndim != 2:
                    mask = mask[:, :, 0]
                mask = mask / 255
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                X[batch_index] = img
                Y[batch_index] = np.reshape(mask, mask.shape + (1,))
            yield X, Y


class SegmLoss(Loss):

    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='segm_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        gt = gt[:, 94:-127, 94:-127, :]
        pr = pr[:, 94:-127, 94:-127, :]
        backend = self.submodules['backend']

        sum_gt = backend.mean(gt, axis=(-3, -2, -1))
        inv_sum_gt = 1 - sum_gt
        width = tf.shape(gt)[2]
        height = tf.shape(gt)[1]
        size = tf.cast(width * height, tf.float32)

        coef1 = 1.0
        coef0 = 1.0
        intersect = coef1 * backend.sum(gt * pr, axis=(-3, -2, -1)) / size
        inv_intersect = coef0 * backend.sum((1 - gt) * (1 - pr), axis=(-3, -2, -1)) / size
        loss = (coef1 * sum_gt + coef0 * inv_sum_gt) - intersect - inv_intersect
        return loss


segm_loss = SegmLoss()


def segm_loss_np(gt, pr):
    gt = gt[:, 94:-127, 94:-127, :]
    pr = pr[:, 94:-127, 94:-127, :]
    sum_gt = np.mean(gt, axis=(-3, -2, -1))
    inv_sum_gt = 1 - sum_gt
    width = gt.shape[2]
    height = gt.shape[1]
    size = width * height
    coef1 = 1.0
    coef0 = 1.0
    intersect = coef1 * np.sum(gt * pr, axis=(-3, -2, -1)) / size
    inv_intersect = coef0 * np.sum((1 - gt) * (1 - pr), axis=(-3, -2, -1)) / size
    return (coef1 * sum_gt + coef0 * inv_sum_gt) - intersect - inv_intersect


class PreventLossIncrease(Keras.callbacks.Callback):
    def __init__(self, model_file_name, train_gen, step_count, val_gen, val_batch_count, best_loss=np.Inf):
        super(PreventLossIncrease, self).__init__()
        self.best_avg_loss = best_loss
        self.model_file_name = model_file_name
        self.train_gen = train_gen
        self.step_count = step_count
        self.val_gen = val_gen
        self.val_batch_count = val_batch_count

    def on_train_begin(self, logs=None):
        self.file = open("fit.log", "w")
        print('segm_loss,val_segm_loss', file=self.file)
        self.file.flush()

    def on_train_end(self, logs=None):
        self.file.close()

    def on_epoch_end(self, epoch, logs=None):
        print()
        tr_loss = 0
        for i in range(self.step_count):
            xy = next(self.train_gen)
            res = self.model.predict(xy[0], verbose=0)
            res_et = xy[1]
            tr_loss += np.mean(segm_loss_np(res_et, res))
            if i % 10 == 0:
                print('|', end='', flush=True)
        tr_loss /= self.step_count
        print(' loss:', tr_loss)
        val_loss = 0
        for i in range(self.val_batch_count):
            xy = next(self.val_gen)
            res = self.model.predict(xy[0], verbose=0)
            res_et = xy[1]
            val_loss += np.mean(segm_loss_np(res_et, res))
            if i % 10 == 0:
                print('|', end='', flush=True)
        val_loss /= self.val_batch_count
        print(' val_loss:', val_loss)
        print(tr_loss, ',', val_loss, sep='', file=self.file)
        self.file.flush()
        loss = val_loss
        avg_loss = 0.8 * loss + 0.2 * tr_loss
        print('avg_loss =', avg_loss)
        if avg_loss < self.best_avg_loss:
            self.best_avg_loss = avg_loss
            self.model.save(self.model_file_name)
            print('Model saved')


def main():
    TileSize = 512
    train_data_dir = '../aviris/tiles/edge_set/train/'
    test_data_dir = '../aviris/tiles/edge_set/test/'
    model_file_name = 'unet_d5_ch16_mp_1conv_bn_ms_plus_edges'
    is_need_load_model = True

    batch_size = 12

    steps_per_epoch = math.floor(len(glob.glob1(train_data_dir + 'mask/', "*.tif")) / batch_size)
    steps_val = math.floor(len(glob.glob1(test_data_dir + 'mask/', "*.tif")) / batch_size)

    myGene = train_generator(train_data_dir, 'img', 'mask', 0, steps_per_epoch * batch_size, batch_size, TileSize)
    myGene1 = train_generator(train_data_dir, 'img', 'mask', 0, steps_per_epoch * batch_size, batch_size, TileSize)
    valGene = train_generator(test_data_dir, 'img', 'mask', 0, steps_val * batch_size, batch_size, TileSize)

    if is_need_load_model and os.path.isfile(model_file_name + '.hdf5'):
        model = load_model(model_file_name + '.hdf5', compile=True, custom_objects={'segm_loss': segm_loss})

        # for layer in model.layers:
        #    if type(layer) is BatchNormalization:
        #        layer.momentum = 0.999
        # model.save('tmp.hdf5')
        # model = load_model('tmp.hdf5', compile=True, custom_objects={'segm_loss': segm_loss})

        # Keras.backend.set_value(model.optimizer.lr, 1e-5)
    else:
        model = UNet(img_shape=(TileSize, TileSize, 4), out_ch=1, start_ch=16, depth=5, inc_rate=2, dropout=0,
                     batchnorm=True, maxpool=True, upconv=True, residual=False, second_conv=False)
        print(model.summary())
        model.compile(optimizer=Adam(lr=1e-4), loss=segm_loss)

    model.fit_generator(myGene, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[
        PreventLossIncrease(model_file_name + '.hdf5', myGene1, steps_per_epoch, valGene, steps_val)])
    # model.save(model_file_name + '.hdf5')
    # Keras2Tensorflow(model, model_file_name + '.pb')
