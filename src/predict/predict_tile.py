import numpy as np
from tensorflow.keras.models import load_model
from src.traning import segm_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from skimage import io as io
import matplotlib.pyplot as plt

IMG_PATH = '../../aviris/tiles/rgb_set/test/img/47.tif'
MASK_PATH = '../../aviris/tiles/rgb_set/test/mask/47.tif'
LEFT_TOP_OFFSET = 94
RIGHT_BOTTOM_OFFSET = 127
TILE_SIZE = 512
MAX_BRIGHT_CODE = 255


def predict_tile(tile, model):
    tile_tmp = np.pad(tile, ((LEFT_TOP_OFFSET, RIGHT_BOTTOM_OFFSET), (LEFT_TOP_OFFSET, RIGHT_BOTTOM_OFFSET), (0, 0)),
                      'constant')
    rows, cols, dims = tile_tmp.shape
    xstop, ystop = False, False
    ypos_tmp, ypos = 0, 0
    step = TILE_SIZE - LEFT_TOP_OFFSET - RIGHT_BOTTOM_OFFSET
    unet_input = np.zeros((1, TILE_SIZE, TILE_SIZE, dims))
    out = np.zeros((TILE_SIZE, TILE_SIZE)).astype(np.uint8)
    while not ystop:
        if ypos_tmp + TILE_SIZE > rows:
            ypos_tmp = rows - TILE_SIZE
            ypos = TILE_SIZE - step
            ystop = True
        xpos_tmp, xpos = 0, 0
        while not xstop:
            if xpos_tmp + TILE_SIZE > cols:
                xpos_tmp = cols - TILE_SIZE
                xpos = TILE_SIZE - step
                xstop = True
            cur_tile = tile_tmp[ypos_tmp:ypos_tmp + TILE_SIZE, xpos_tmp:xpos_tmp + TILE_SIZE] / MAX_BRIGHT_CODE
            unet_input[0] = cur_tile
            unet_out = model.predict(unet_input)
            prediction = norm_prediction(unet_out[0])
            prediction = prediction[LEFT_TOP_OFFSET:-RIGHT_BOTTOM_OFFSET, LEFT_TOP_OFFSET:-RIGHT_BOTTOM_OFFSET, -1]
            out[ypos:ypos + step, xpos:xpos + step] = prediction
            xpos += step
            xpos_tmp += TILE_SIZE
        ypos += step
        ypos_tmp += TILE_SIZE
        if xstop:
            xstop = False
    return out


def norm_prediction(pred):
    pred[pred > 0.5] = 255
    pred[pred <= 0.5] = 0
    return pred


def calc_metrics(ytrue, ypred):
    ytrue = np.array(ytrue).astype(np.uint8).flatten()
    ypred = np.array(ypred).astype(np.uint8).flatten()
    pr, rc, fs, sp = precision_recall_fscore_support(ytrue, ypred, labels=[0, 255])
    print(f'Precision: {str(pr)} , Recall: {str(rc)} , F-Score: {str(fs)} , Support: {str(sp)}.')


def main():
    img = io.imread(IMG_PATH)
    mask = io.imread(MASK_PATH)
    if img.shape[0] != TILE_SIZE or img.shape[1] != TILE_SIZE:
        print(f'Image size must be {str(TILE_SIZE)} !!!')
        return
    model_file_name = '../unet_d5_ch16_mp_1conv_bn_ms.hdf5'
    model = load_model(model_file_name, compile=True, custom_objects={'segm_loss': segm_loss})
    pr = predict_tile(img, model)
    calc_metrics(pr, mask)
    plt.figure(0), plt.imshow(img), plt.title('IMG')
    plt.figure(1), plt.imshow(pr, cmap='gray'), plt.title('PREDICT')
    plt.figure(2), plt.imshow(mask, cmap='gray'), plt.title('MASK')
    plt.show()

if __name__ == '__main__':
    main()
