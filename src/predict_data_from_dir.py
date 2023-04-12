import skimage.io as io
from make_rgb_dataset import read_data_from_dir, save_img_set_to_dir
import numpy as np
from tensorflow.keras.models import load_model
from traning import segm_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

IMG_DIR = '../aviris/tiles/rgb_set/test/img/'
FILE_EXT = '.tif'
TILE_SIZE = 256
DIMS = 3
MAX_BRIGHT_CODE = 255
USING_GTRUTH = True
MASK_DIR = '../aviris/tiles/rgb_set/test/mask/'


def tiles_gen(input_tiles):
    out_tiles = np.zeros((len(input_tiles), TILE_SIZE, TILE_SIZE, DIMS))
    for i in range(len(input_tiles)):
        out_tiles[i] = input_tiles[i] / MAX_BRIGHT_CODE
    yield out_tiles


def norm_predictions(predictions_set):
    out_tiles = []
    for item in enumerate(predictions_set):
        item = item[1]
        item[item > 0.5] = 255
        item[item <= 0.5] = 0
        out_tiles.append(item)
    return out_tiles


def calc_metrics(ytrue, ypred):
    ytrue = np.array(ytrue).astype(np.uint8).flatten()
    ypred = np.array(ypred).astype(np.uint8).flatten()
    tp, fn, fp, tn = confusion_matrix(ytrue, ypred, labels=[0, 255]).ravel()
    pr, rc, fs, sp = precision_recall_fscore_support(ytrue, ypred, labels=[0, 255])
    return [tp, fn, fp, tn], [pr, rc, fs, tn]


def calc_metrics_for_set(ytrue_set, ypred_set):
    gl_tp, gl_fn, gl_fp, gl_tn = 0, 0, 0, 0
    gl_pr, gl_rc, gl_fs, gl_sp = 0, 0, 0, 0
    for i in range(len(ytrue_set)):
        confmat, metrics = calc_metrics(ytrue_set[i], ypred_set[i])
        gl_tp += confmat[0] / len(ytrue_set)
        gl_fn += confmat[1] / len(ytrue_set)
        gl_fp += confmat[2] / len(ytrue_set)
        gl_tn += confmat[3] / len(ytrue_set)
        gl_pr += metrics[0] / len(ytrue_set)
        gl_rc += metrics[1] / len(ytrue_set)
        gl_fs += metrics[2] / len(ytrue_set)
        gl_sp += metrics[3] / len(ytrue_set)
    print('Averege confusion matrix: ')
    print('True positive: ' + str(gl_tp)
          + ' False negative: ' + str(gl_fn)
          + ' False positive: ' + str(gl_fp)
          + ' True negative: ' + str(gl_tn))
    print('Average metrics values: ')
    print(' Precision: ' + str(gl_pr)
          + ' Recall: ' + str(gl_rc)
          + ' FScore: ' + str(gl_fs))


def main():
    img_set = read_data_from_dir(IMG_DIR, FILE_EXT)
    tiles = tiles_gen(img_set)
    model_file_name = '../unet_d5_ch16_mp_1conv_bn_ms.hdf5'
    model = load_model(model_file_name, compile=True, custom_objects={'segm_loss': segm_loss})
    predictions = model.predict_generator(tiles, 1, verbose=1)
    predictions = norm_predictions(predictions)
    out_dir = '../tmp/'
    save_img_set_to_dir(out_dir, predictions, FILE_EXT)
    if USING_GTRUTH:
        mask_set = read_data_from_dir(MASK_DIR, FILE_EXT)
        calc_metrics_for_set(mask_set, predictions)


if __name__ == '__main__':
    main()
