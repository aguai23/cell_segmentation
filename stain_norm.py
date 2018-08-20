import openslide
from staintools import MacenkoNormalizer
from staintools.utils.visual import read_image, show
import scipy.misc as mc
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage.morphology import dilation, erosion, square
dir = "/home/yunzhe/Downloads/segmentation_training_set/"
source_file = "/home/yunzhe/Downloads/segmentation_training_set/image02.png"
save_dir = "/home/yunzhe/norm_data_new/"
source = read_image(source_file)
normalizer = MacenkoNormalizer()
normalizer.fit(source)
filter = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
for filename in os.listdir(dir):
    if filename.endswith(".png") and not filename.endswith("_poly.png"):
        target = read_image(dir + filename)
        if dir + filename == source_file:
            mc.imsave(save_dir + filename, target)
        else:
            norm_target = normalizer.transform(target)
            show(norm_target, fig_size=(4,4))
            mc.imsave(save_dir + filename, norm_target)

        filename_suffix = filename.split(".")[0]
        mask_file = dir + filename_suffix + "_mask.txt"
        if os.path.exists(mask_file):
            width = 0
            height = 0
            with open(mask_file) as file:
                lines = file.readlines()
                index = 0
                mask = np.zeros((target.shape[0], target.shape[1]), np.int32)
                contour = np.zeros(mask.shape, np.int32)
                for line in lines:
                    if width == 0:
                        width, height = int(line.split()[0]), int(line.split()[1])
                        assert width * height == len(lines) - 1
                    else:
                        x = int(index / width)
                        y = int(index % width)
                        mask[x][y] = int(line)
                        index += 1
                region_props = measure.regionprops(mask)
                for region in region_props:
                    temp_mask = np.zeros(mask.shape)
                    temp_mask[np.where(mask == region.label)] = 1
                    dilate_mask = dilation(temp_mask, filter)
                    erode_mask = erosion(temp_mask, filter)
                    contour = np.logical_or(contour, dilate_mask - erode_mask)

                plt.imshow(contour)
                plt.show()
                plt.imshow(mask)
                plt.show()
                # mc.imsave(save_dir + filename_suffix + ".png", mask)
                np.save(save_dir + filename_suffix + ".npy", mask)
                np.save(save_dir + filename_suffix + "_contour.npy", contour)

