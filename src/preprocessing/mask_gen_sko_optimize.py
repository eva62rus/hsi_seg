import math

import numpy as np
import tkinter
from tkinter import filedialog
from skimage import io


def main():
    file_dlg = tkinter.Tk()
    file_dlg.withdraw()

    img_path = filedialog.askopenfilename()
    img = io.imread(img_path)
    rows, cols = img.shape

    print("Source image: " + img_path)
    print(img.shape)

    window_size = 9
    no_pix_values = window_size // 2
    tmp_img = np.pad(img, no_pix_values, 'reflect').astype(np.uint8)
    tmp_rows, tmp_cols = tmp_img.shape
    print(tmp_img.shape)
    pre_proc_sko_mask = np.zeros((rows, cols)).astype(np.uint8)
    sko_thresh = 30

    y = 0
    for i in range(no_pix_values, tmp_rows - no_pix_values):
        print("Processing " + str(y) + " row...")
        x = 0
        for j in range(no_pix_values, tmp_cols - no_pix_values):
            window_values = tmp_img[i - no_pix_values: (i - no_pix_values) + window_size,
                            j - no_pix_values: (j - no_pix_values) + window_size]
            window_sum = np.sum(window_values)
            window_avg = window_sum / (window_size * window_size)
            dispersion = pow(window_values - window_avg, 2)
            sko = math.sqrt(np.sum(dispersion) / (window_size * window_size))
            if sko <= sko_thresh:
                pre_proc_sko_mask[y][x] = 255
            x += 1
        y += 1

    io.imsave("rms_mask.tif", pre_proc_sko_mask, check_contrast=False)

    print("RMS mask calculated is finish.")
    print("Post processing is started...")

    y = 0
    bright_thresh = 30
    tmp_sko_mask = np.pad(pre_proc_sko_mask, no_pix_values, 'reflect').astype(np.uint8)
    post_proc_sko_mask = np.zeros((rows, cols)).astype(np.uint8)
    for i in range(no_pix_values, tmp_rows - no_pix_values):
        print("Post processing " + str(y) + " row...")
        x = 0
        for j in range(no_pix_values, tmp_cols - no_pix_values):
            if pre_proc_sko_mask[y][x] == 255:
                post_proc_sko_mask[y][x] = 255
            else:
                img_win_values = tmp_img[i - no_pix_values: (i - no_pix_values) + window_size,
                                 j - no_pix_values: (j - no_pix_values) + window_size]
                mask_win_values = tmp_sko_mask[i - no_pix_values: (i - no_pix_values) + window_size,
                                  j - no_pix_values: (j - no_pix_values) + window_size]
                fill_pix_count = mask_win_values[mask_win_values != 0].size
                sum_mask = np.array(mask_win_values).astype(np.bool)
                sum_fill_pix = np.sum(img_win_values, where=sum_mask)
                if sum_fill_pix > 0:
                    avg_fill_pix = sum_fill_pix / fill_pix_count
                    if abs(avg_fill_pix - img[y][x]) <= bright_thresh:
                        post_proc_sko_mask[y][x] = 255
            x += 1
        y += 1

    io.imsave("rms_mask_edge.tif", post_proc_sko_mask, check_contrast=False)


if __name__ == '__main__':
    main()
