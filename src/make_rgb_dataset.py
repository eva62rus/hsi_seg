from skimage import io
import os
from matplotlib import pyplot as plt
import random

SRC_IMG_DIR = '../aviris/src/img/rgb/'
SRC_MASK_DIR = '../aviris/src/mask/'
SRC_IMG_EXT = '.bmp'
SRC_MASK_EXT = '.tif'
OUT_TEST_IMG_DIR = '../aviris/tiles/rgb_set/test/img/'
OUT_TEST_MASK_DIR = '../aviris/tiles/rgb_set/test/mask/'
OUT_TRAIN_IMG_DIR = '../aviris/tiles/rgb_set/train/img/'
OUT_TRAIN_MASK_DIR = '../aviris/tiles/rgb_set/train/mask/'

OUT_IMG_EXT = '.tif'
OUT_MASK_EXT = '.tif'
TILE_SIZE = 256
OVERLAPPING = 0.75
TEST_PERCENT = 0.05


def display_tile(img, mask):
    plt.figure(1)
    dims = img.shape
    if len(dims) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.figure(2)
    plt.imshow(mask, cmap='gray')
    plt.show()


def display_tile_from_drive(tile_number, test_data=False):
    if test_data:
        img = io.imread(OUT_TEST_IMG_DIR + str(tile_number) + OUT_IMG_EXT)
        mask = io.imread(OUT_TEST_MASK_DIR + str(tile_number) + OUT_MASK_EXT)
    else:
        img = io.imread(OUT_TRAIN_IMG_DIR + str(tile_number) + OUT_IMG_EXT)
        mask = io.imread(OUT_TRAIN_MASK_DIR + str(tile_number) + OUT_MASK_EXT)
    display_tile(img, mask)


def print_source_data_location():
    print('Source data location: ')
    print('Images: ' + SRC_IMG_DIR)
    print('Masks: ' + SRC_MASK_DIR)


def read_data_from_dir(path2dir, file_ext):
    file_names = os.listdir(path2dir)
    file_names_sort = [str(i) + '.tif' for i in range(len(file_names))]
    out_data = []
    for file_name in file_names:
        if file_name[-len(file_ext):] == file_ext:
            out_data.append(io.imread(path2dir + file_name))
    return out_data


# overlapping - [0..1]
def cropping_img_set(input_img_set, tile_size, overlapping):
    shift = int(tile_size * overlapping)
    xbreak = False
    ybreak = False
    out_tiles_set = []
    for cur_img in input_img_set:
        rows = cur_img.shape[0]
        cols = cur_img.shape[1]
        ypos = 0
        while not ybreak:
            if ypos + tile_size > rows:
                ypos = rows - tile_size
                ybreak = True
            xpos = 0
            while not xbreak:
                if xpos + tile_size > cols:
                    xpos = cols - tile_size
                    xbreak = True
                tile = cur_img[ypos:ypos + tile_size, xpos:xpos + tile_size]
                out_tiles_set.append(tile)
                xpos += shift
            ypos += shift
            if xbreak:
                xbreak = False
        if ybreak:
            ybreak = False
    return out_tiles_set


# test_percent - [0..1]
def split_by_train_and_test(input_img_tiles, input_mask_tiles, test_percent):
    train_images, train_masks, test_images, test_masks = [], [], [], []
    total_test_tiles = int(len(input_img_tiles) * test_percent)
    test_freq = int(len(input_img_tiles) / total_test_tiles)
    for i in range(len(input_img_tiles)):
        if i % test_freq == 0:
            test_images.append(input_img_tiles[i])
            test_masks.append(input_mask_tiles[i])
        else:
            train_images.append(input_img_tiles[i])
            train_masks.append(input_mask_tiles[i])
    print('Traning tiles count: ' + str(len(train_images)))
    print('Testing tiles count: ' + str(len(test_images)))
    return train_images, train_masks, test_images, test_masks


def mix_dataset(input_img_set, input_mask_set):
    out_img_set, out_mask_set = [], []
    new_indeces = [i for i in range(len(input_img_set))]
    random.shuffle(new_indeces)
    for new_index in new_indeces:
        out_img_set.append(input_img_set[new_index])
        out_mask_set.append(input_mask_set[new_index])
    return out_img_set, out_mask_set


def save_img_set_to_dir(path2dir, input_img_set, file_ext):
    file_names = [path2dir + str(i) + file_ext for i in range(len(input_img_set))]
    for i in range(len(input_img_set)):
        io.imsave(file_names[i], input_img_set[i], check_contrast=False)


def main():
    print_source_data_location()

    src_images = read_data_from_dir(SRC_IMG_DIR, SRC_IMG_EXT)
    src_masks = read_data_from_dir(SRC_MASK_DIR, SRC_MASK_EXT)

    img_tiles = cropping_img_set(src_images, TILE_SIZE, OVERLAPPING)
    mask_tiles = cropping_img_set(src_masks, TILE_SIZE, OVERLAPPING)
    print('Total tiles count: ' + str(len(img_tiles)))

    train_img_set, train_mask_set, test_img_set, test_mask_set \
        = split_by_train_and_test(img_tiles, mask_tiles, TEST_PERCENT)

    test_img_set, test_mask_set = mix_dataset(test_img_set, test_mask_set)

    save_img_set_to_dir(OUT_TEST_IMG_DIR, test_img_set, OUT_IMG_EXT)
    save_img_set_to_dir(OUT_TEST_MASK_DIR, test_mask_set, OUT_MASK_EXT)

    train_img_set, train_mask_set = mix_dataset(train_img_set, train_mask_set)

    save_img_set_to_dir(OUT_TRAIN_IMG_DIR, train_img_set, OUT_IMG_EXT)
    save_img_set_to_dir(OUT_TRAIN_MASK_DIR, train_mask_set, OUT_MASK_EXT)


if __name__ == '__main__':
    main()
