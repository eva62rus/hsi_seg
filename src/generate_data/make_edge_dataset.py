from src.generate_data.make_rgb_dataset import *

SRC_IMG_DIR = '../../aviris/src/img/rgb/'
SRC_MASK_DIR = '../../aviris/src/mask/'
SRC_EDGE_DIR = '../../aviris/src/edges/'

SRC_IMG_EXT = '.bmp'
SRC_MASK_EXT = '.tif'
SRC_EDGE_EXT = '.bmp'

OUT_TEST_IMG_DIR = '../../aviris/tiles/edge_set/test/img/'
OUT_TEST_MASK_DIR = '../../aviris/tiles/edge_set/test/mask/'
OUT_TRAIN_IMG_DIR = '../../aviris/tiles/edge_set/train/img/'
OUT_TRAIN_MASK_DIR = '../../aviris/tiles/edge_set/train/mask/'


def read_src_data_from_dir(path2dir, file_ext):
    file_names = os.listdir(path2dir)
    out_data = []
    for file_name in file_names:
        if file_name[-len(file_ext):] == file_ext:
            out_data.append(io.imread(path2dir + file_name))
    return out_data


def apply_edge2img(img, edge):
    rows, cols, dims = img.shape
    out_img = np.zeros((rows, cols, dims + 1)).astype(np.uint8)
    out_img[:, :, 0:3] = img
    out_img[:, :, 3] = edge
    return out_img


def main():
    src_images = read_src_data_from_dir(SRC_IMG_DIR, SRC_IMG_EXT)
    src_masks = read_src_data_from_dir(SRC_MASK_DIR, SRC_MASK_EXT)
    src_edges = read_src_data_from_dir(SRC_EDGE_DIR, SRC_EDGE_EXT)

    new_images = []
    for src_img, src_edge in zip(src_images, src_edges):
        new_images.append(apply_edge2img(src_img, src_edge))

    for i in range(len(new_images)):
        new_images[i], src_masks[i] = supplement_sample(new_images[i], src_masks[i])

    img_tiles = cropping_img_set(new_images, TILE_SIZE, OVERLAPPING)
    mask_tiles = cropping_img_set(src_masks, TILE_SIZE, OVERLAPPING)

    print('Total tiles count: ' + str(len(img_tiles)))

    train_img_set, train_mask_set, test_img_set, test_mask_set \
        = split_by_train_and_test(img_tiles, mask_tiles, TEST_PERCENT)

    save_img_set_to_dir(OUT_TEST_IMG_DIR, test_img_set, OUT_IMG_EXT)
    save_img_set_to_dir(OUT_TEST_MASK_DIR, test_mask_set, OUT_MASK_EXT)

    train_img_set, train_mask_set = mix_dataset(train_img_set, train_mask_set)

    save_img_set_to_dir(OUT_TRAIN_IMG_DIR, train_img_set, OUT_IMG_EXT)
    save_img_set_to_dir(OUT_TRAIN_MASK_DIR, train_mask_set, OUT_MASK_EXT)


if __name__ == '__main__':
    main()
