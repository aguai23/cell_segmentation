import numpy as np
import openslide
import os
import xml.etree.ElementTree as ET
from matplotlib.path import Path
from matplotlib import pyplot as plt
import scipy.misc as mc
import random
from sklearn.utils import shuffle
from PIL import Image


class SegmentationDataProvider:

    def __init__(self, data_dir, train_percent=0.7, sample_size=224, num_class=2, shuffle_data=True):
        self.train_data = []
        self.valid_data = []
        self.train_mask = []
        self.train_contour_mask = []
        self.valid_mask = []
        self.valid_contour_mask = []
        self.index = -1
        self.sample_size = sample_size
        self.num_class = num_class
        filenames = []
        # read data from file
        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                filenames.append(filename)

        self.train_size = int(train_percent * len(filenames))
        self.valid_size = len(filenames) - self.train_size

        count = 0
        valid_limit = 9
        for filename in filenames:

            filename_suffix = filename.split(".")[0]
            image = mc.imread(data_dir + filename)
            mask = np.load(data_dir + filename_suffix + ".npy")
            contour_mask = np.load(data_dir + filename_suffix + "_contour.npy")

            if count < self.train_size:
                self.sample_data(image, mask, contour_mask, True, sample_size)
            elif count < self.train_size + valid_limit:
                self.sample_data(image, mask, contour_mask, False, sample_size)
            count += 1
        print(len(self.train_data))
        print(len(self.valid_data))

        if shuffle_data:
            shuffled_train_data, shuffled_train_mask, shuffled_train_contour = shuffle(self.train_data,
                                                                                       self.train_mask,
                                                                                       self.train_contour_mask,
                                                                                      )
            self.train_data = shuffled_train_data
            self.train_mask = shuffled_train_mask
            self.train_contour_mask = shuffled_train_contour

    def __call__(self, size):

        X = np.zeros((size, self.sample_size, self.sample_size, 3))
        Y = np.zeros((size, self.sample_size, self.sample_size, 2))
        Y_contour = np.zeros((size, self.sample_size, self.sample_size, 2))

        for i in range(size):
            self.index += 1
            if self.index >= self.train_size:
                self.index = 0
            X[i] = self.train_data[self.index]
            Y[i] = self.train_mask[self.index]
            Y_contour[i] = self.train_contour_mask[self.index]

        return X, Y, Y_contour

    def sample_data(self, image, mask, contour_mask, train, sample_size):
        width, height, _ = image.shape
        for i in range(100):
            x = random.randint(0, width - sample_size)
            y = random.randint(0, height - sample_size)
            image_crop = image[x:x + sample_size, y:y + sample_size, 0:3]
            mask_crop = self.convert_to_onehot(mask[x:x + sample_size, y:y + sample_size])
            contour_mask_crop = self.convert_to_onehot(contour_mask[x:x + sample_size, y:y + sample_size])
            if train:
                self.train_data.append(image_crop)
                self.train_mask.append(mask_crop)
                self.train_contour_mask.append(contour_mask_crop)
            else:
                self.valid_data.append(image_crop)
                self.valid_mask.append(mask_crop)
                self.valid_contour_mask.append(contour_mask_crop)

    def verification_data(self):
        return self.valid_data, self.valid_mask, self.valid_contour_mask

    def convert_to_onehot(self, array):
        result = np.zeros((array.shape[0], array.shape[1], self.num_class), dtype=np.float32)
        for i in range(self.num_class):
            layer = np.zeros(array.shape)
            layer[np.where(array == i)] = 1
            result[..., i] = layer
        return result


def find_contour(add_mask):
    contour_mask = np.zeros(mask.shape)
    width, height = add_mask.shape
    for i in range(0, width):
        for j in range(0, height):
            for inner_i in range(-2, 3):
                for inner_j in range(-2, 3):
                    if (0 <= i + inner_i < width) and (0 <= j + inner_j < height):
                        if add_mask[i + inner_i][j + inner_j] != add_mask[i][j]:
                            contour_mask[i][j] = 1

    return contour_mask


if __name__ == "__main__":

    data_dir = "/home/yunzhe/Downloads/MoNuSeg Training Data/MoNuSeg Training Data/Tissue images/"
    annotation_dir = "/home/yunzhe/Downloads/MoNuSeg Training Data/MoNuSeg Training Data/Annotations/"
    save_dir = "/home/yunzhe/Downloads/MoNuSeg Training Data/training_data/"

    for filename in os.listdir(data_dir):
        print(filename)
        if filename.endswith(".tif"):

            # read image file
            file_object = openslide.open_slide(data_dir + filename)
            width, height = file_object.dimensions
            image = file_object.read_region((0, 0), 0, (width, height))
            image_array = np.asarray(image)[:, :, 0:3]
            file_object.close()

            # read xml file
            filename_suffix = filename.split(".")[0]
            annotation_file = annotation_dir + filename_suffix + ".xml"
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            mask = np.zeros((width, height))
            add_mask = np.zeros((width, height))

            # extract each nuclear
            for region in root.iter("Region"):
                polygon = []
                for vertex in region.iter("Vertex"):
                    polygon.append((float(vertex.attrib['X']), float(vertex.attrib['Y'])))
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                path = Path(polygon)
                grid = path.contains_points(points)
                grid = grid.reshape((width, height))

                mask = np.logical_or(grid, mask)
                add_mask = np.add(add_mask, grid)

            # save the result for training
            contour_mask = find_contour(add_mask)
            image.save(save_dir + filename_suffix + ".png")
            np.save(save_dir + filename_suffix, mask)
            np.save(save_dir + filename_suffix + "_contour", contour_mask)
            plt.imshow(contour_mask)
            plt.show()
            plt.imshow(mask)
            plt.show()
