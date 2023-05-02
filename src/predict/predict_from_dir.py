import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from src.traning import segm_loss
from src.generate_data.make_rgb_dataset import read_data_from_dir, save_img_set_to_dir
from predict_tile import predict_tile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from skimage import io as io
from datetime import datetime, time

IMG_FOLDER_PATH = '../../aviris/tiles/edge_set/test/img/'
MASK_FOLDER_PATH = '../../aviris/tiles/edge_set/test/mask/'
FILE_EXT = '.tif'


def calc_materics(ytrue, ypred, outfilename, tile_id):
    ytrue = np.array(ytrue).astype(np.uint8).flatten()
    ypred = np.array(ypred).astype(np.uint8).flatten()
    pr, rc, fs, sp = precision_recall_fscore_support(ytrue, ypred, labels=[0, 255])
    with open(outfilename, 'a') as f:
        f.write(f'Date: {datetime.today().strftime("%d-%m-%Y %H:%M:%S")} \n')
        f.write(f'Tile id: {tile_id} \n')
        f.write(f'Precision: {str(pr)}, Recall: {str(rc)}, FScore: {str(fs)}, Support: {str(sp)} \n\n\n')
    return pr, rc, fs


def main():
    model_file_name = '../unet_d5_ch16_mp_1conv_bn_ms_plus_edges.hdf5'
    model = load_model(model_file_name, compile=True, custom_objects={'segm_loss': segm_loss})
    img_set = read_data_from_dir(IMG_FOLDER_PATH, FILE_EXT)
    mask_set = read_data_from_dir(MASK_FOLDER_PATH, FILE_EXT)
    predictions_set = []
    # precision, recall and F-score respectively
    total_metrics = [0, 0, 0]
    for i in range(len(img_set)):
        pred = predict_tile(img_set[i], model)
        predictions_set.append(pred)
        metrics = calc_materics(mask_set[i], pred, 'out_metrics.txt', i)
        for j in range(len(metrics)):
            total_metrics[j] += metrics[j] / len(img_set)
    save_img_set_to_dir(path2dir='../../tmp/', input_img_set=predictions_set, file_ext=FILE_EXT)
    print(f'Global values: \n Precision: {str(total_metrics[0])} \n'
          f' Recall: {str(total_metrics[1])} \n FScore: {str(total_metrics[2])}')


if __name__ == '__main__':
    main()
