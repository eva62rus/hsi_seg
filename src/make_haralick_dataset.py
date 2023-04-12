import numpy as np
from skimage import io
import os
import random
from make_rgb_dataset import *
from scipy.io import savemat

SRC_IMG_DIR = '../aviris/src/img/rgb/'
SRC_MASK_DIR = '../aviris/src/mask/'
SRC_TXT_DIR = '../aviris/src/texture/'
SRC_IMG_EXT = '.bmp'
SRC_MASK_EXT = '.tif'
SRC_TXT_EXT = '.bmp'
OUT_TEST_IMG_DIR = '../aviris/tiles/haralick_set/test/img/'
OUT_TEST_MASK_DIR = '../aviris/tiles/haralick_set/test/mask/'
OUT_TRAIN_IMG_DIR = '../aviris/tiles/haralick_set/train/img/'
OUT_TRAIN_MASK_DIR = '../aviris/tiles/haralick_set/train/mask/'
OUT_IMG_EXT = '.tif'
OUT_MASK_EXT = '.tif'
TILE_SIZE = 256
OVERLAPPING = 1
TEST_PERCENT = 0.1


def apply_texture2img(img, texture):
    rows, cols, dims = img.shape
    out_img = np.zeros((rows, cols, dims + 1)).astype(np.uint8)
    out_img[:, :, 0:3] = img
    out_img[:, :, 3] = texture
    return out_img


def save_img4d_set_to_dir(path2dir, input_img_set):
    file_names = [path2dir + str(i) + '.mat' for i in range(len(input_img_set))]
    for i in range(len(input_img_set)):
        cur_img = {'img': input_img_set[i]}
        savemat(file_names[i], cur_img)


def main():
    src_images = read_data_from_dir(SRC_IMG_DIR, SRC_IMG_EXT)
    src_masks = read_data_from_dir(SRC_MASK_DIR, SRC_MASK_EXT)
    src_textures = read_data_from_dir(SRC_TXT_DIR, SRC_TXT_EXT)

    new_images = []
    for i in range(len(src_images)):
        new_images.append(apply_texture2img(src_images[i], src_textures[i]))

    img_tiles = cropping_img_set(new_images, TILE_SIZE, OVERLAPPING)
    mask_tiles = cropping_img_set(src_masks, TILE_SIZE, OVERLAPPING)

    print('Total tiles count: ' + str(len(img_tiles)))

    train_img_set, train_mask_set, test_img_set, test_mask_set \
        = split_by_train_and_test(img_tiles, mask_tiles, TEST_PERCENT)

    save_img4d_set_to_dir(OUT_TEST_IMG_DIR, test_img_set)
    save_img_set_to_dir(OUT_TEST_MASK_DIR, test_mask_set, OUT_MASK_EXT)

    train_img_set, train_mask_set = mix_dataset(train_img_set, train_mask_set)

    save_img4d_set_to_dir(OUT_TRAIN_IMG_DIR, train_img_set)
    save_img_set_to_dir(OUT_TRAIN_MASK_DIR, train_mask_set, OUT_MASK_EXT)


if __name__ == '__main__':
    main()
