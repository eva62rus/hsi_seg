from osgeo import gdal
import numpy as np
import skimage.io as io

FILE_PATH = 'img36.tiff'


def open_raster(p_file_name):
    return gdal.Open(p_file_name)


def get_shape(p_raster):
    return p_raster.RasterYSize, p_raster.RasterXSize, p_raster.RasterCount


def print_raster_info(p_raster):
    print('YSize: ' + str(p_raster.RasterYSize) + '\n' +
          'XSize: ' + str(p_raster.RasterXSize) + '\n' +
          'NBands: ' + str(p_raster.RasterCount))


def get_band(p_raster, p_band_num):
    return p_raster.GetRasterBand(p_band_num).ReadAsArray()


def save_bands_to_tiff(p_filename, p_raster, p_rows, p_cols,
                       p_first_band_num, p_last_band_num):
    out_nbands = p_last_band_num - p_first_band_num
    driver = gdal.GetDriverByName("GTiff")
    out_data = driver.Create(p_filename, p_cols, p_rows, out_nbands, gdal.GDT_UInt16)
    band_num = 1
    for i in range(p_first_band_num, p_last_band_num):
        out_data.GetRasterBand(band_num).WriteArray(get_band(p_raster, i))
        band_num += 1
    out_data.FlushCache()


def main():
    raster = open_raster(FILE_PATH)
    rows, cols, n_bands = get_shape(raster)
    print_raster_info(raster)

    panchrome = np.zeros((rows, cols))
    for band_num in range(10, 65):
        cur_band = get_band(raster, band_num) / n_bands
        panchrome += cur_band

    panchrome = np.array(panchrome, dtype=np.uint64)

    io.imsave('panchrome_all_bands.tif', panchrome.astype(np.uint16), check_contrast=False)
    print('panchrome_all_bands is saved.')

    save_bands_to_tiff('visible_bands.tiff', raster, rows, cols, 10, 65)
    print('multi-bands raster is saved.')


if __name__ == '__main__':
    main()
