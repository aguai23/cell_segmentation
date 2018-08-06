import numpy as np
import openslide
import os
import xml.etree.ElementTree as ET
from matplotlib.path import Path
from matplotlib import pyplot as plt
import scipy.misc as mc
import random
from sklearn.utils import shuffle
from shutil import rmtree
from skimage.morphology import dilation, erosion, square
import cv2
from PIL import Image
import math


class SegmentationDataProvider:

    def __init__(self, data_dir, test_dir, train_percent=0.8, sample_size=224, num_class=2, output_size=219,
                 sample_number=100, shuffle_data=True, resample=False, test=False):
        """
        init method of data provider
        :param data_dir: the data folder containing training and valid data
        :param test_dir: data folder containing test data
        :param train_percent: percentage of training data
        :param sample_size: crop size of training data
        :param num_class: number of classes
        :param sample_number: number of samples from one image
        :param shuffle_data: shuffle the training data
        :param resample: whether to resample the training and validation data
        """
        # array to store the data
        self.train_data = []
        self.valid_data = []
        self.train_mask = []
        self.train_contour_mask = []
        self.valid_mask = []
        self.valid_contour_mask = []

        # image index used to generate train batch
        self.index = -1

        self.output_size = output_size
        self.sample_size = sample_size
        self.num_class = num_class
        self.sample_number = sample_number
        self.train_dir = data_dir
        self.test_dir = test_dir
        train_folder = data_dir + "train/"
        valid_folder = data_dir + "valid/"

        if test:
            return

        if resample or (not os.path.exists(train_folder)):
            # remove previous data
            if os.path.exists(train_folder):
                rmtree(train_folder)
                rmtree(valid_folder)

            # resample the training and valid data
            filenames = []
            # read data from file
            for filename in os.listdir(data_dir):
                if filename.endswith(".png"):
                    filenames.append(filename)

            self.train_size = int(train_percent * len(filenames))
            self.valid_size = len(filenames) - self.train_size

            if not os.path.exists(train_folder):
                os.mkdir(train_folder)
            if not os.path.exists(valid_folder):
                os.mkdir(valid_folder)

            count = 0
            valid_limit = 9
            for filename in filenames:
                filename_suffix = filename.split(".")[0]
                image = mc.imread(data_dir + filename)
                mask = np.load(data_dir + filename_suffix + ".npy")
                contour_mask = np.load(data_dir + filename_suffix + "_contour.npy")

                if count < self.train_size:
                    self.sample_data(filename, image, mask,
                                     contour_mask, True, sample_size, sample_number, train_folder,
                                     output_size=self.output_size)
                elif count < self.train_size + valid_limit:
                    self.sample_data(filename, image, mask,
                                     contour_mask, False, sample_size, sample_number, valid_folder,
                                     output_size=self.output_size)
                count += 1

        else:
            # load saved data
            print("start loading data")
            load_count = 0
            for filename in os.listdir(train_folder):
                if filename.endswith("_image.npy"):
                    image = np.load(train_folder + filename)
                    mask = np.load(train_folder + filename.replace("_image.npy", "_mask.npy"))
                    contour = np.load(train_folder + filename.replace("_image.npy", "_contour.npy"))
                    self.train_data.append(image)
                    self.train_mask.append(mask)
                    self.train_contour_mask.append(contour)
                    load_count += 1
                    if load_count % 10000 == 0:
                        print(load_count)

            for filename in os.listdir(valid_folder):
                if filename.endswith("_image.npy"):
                    image = np.load(valid_folder + filename)
                    mask = np.load(valid_folder + filename.replace("_image.npy", "_mask.npy"))
                    contour = np.load(valid_folder + filename.replace("_image.npy", "_contour.npy"))
                    self.valid_data.append(image)
                    self.valid_mask.append(mask)
                    self.valid_contour_mask.append(contour)

        augment_train_data = []
        augment_train_label = []
        augment_train_contour = []

        for i in range(len(self.train_data)):
            augment_train_data.append(self.train_data[i])
            augment_train_data.append(np.rot90(self.train_data[i], 2))
            augment_train_data.append(np.flip(self.train_data[i], axis=0))
            augment_train_data.append(np.flip(self.train_data[i], axis=1))

            augment_train_label.append(self.train_mask[i])
            augment_train_label.append(np.rot90(self.train_mask[i], 2))
            augment_train_label.append(np.flip(self.train_mask[i], axis=0))
            augment_train_label.append(np.flip(self.train_mask[i], axis=1))

            augment_train_contour.append(self.train_contour_mask[i])
            augment_train_contour.append(np.rot90(self.train_contour_mask[i], 2))
            augment_train_contour.append(np.flip(self.train_contour_mask[i], axis=0))
            augment_train_contour.append(np.flip(self.train_contour_mask[i], axis=1))

        self.train_data = augment_train_data
        self.train_mask = augment_train_label
        self.train_contour_mask = augment_train_contour

        print(len(self.train_data))
        print(len(self.valid_data))
        self.train_size = len(self.train_data)
        self.valid_size = len(self.valid_data)

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
        Y = np.zeros((size, self.output_size, self.output_size, 2))
        Y_contour = np.zeros((size, self.output_size, self.output_size, 2))

        for i in range(size):
            self.index += 1
            if self.index >= self.train_size:
                self.index = 0
            X[i] = self.train_data[self.index]
            Y[i] = self.train_mask[self.index]
            Y_contour[i] = self.train_contour_mask[self.index]

        return X, Y, Y_contour

    def sample_data(self, filename, image, mask, contour_mask, train, sample_size, sample_number, save_folder,
                    output_size=219):
        """
        sample data to generate train and valid data
        :param filename: filename
        :param image: image matrix
        :param mask: mask matrix
        :param contour_mask: contour matrix
        :param train: if it's train
        :param sample_size: sample size
        :param sample_number: number of samples from one image
        :param folder to save
        :return:
        """
        print(filename)
        width, height, _ = image.shape
        image = (image - np.average(image)) / np.std(image)
        binary_mask = np.zeros((width, height))
        binary_mask[np.where(mask >= 1)] = 1
        # binary_mask = erosion(binary_mask, square(5))
        # output_size = ((sample_size + 1) / 2 + 1) / 2
        step = 15
        if not train:
            step = 20
        for i in range(0, width - sample_size, step):
            for j in range(0, height - sample_size, step):
                image_crop = image[i:i + sample_size, j:j + sample_size, 0:3]
                center_i = i + int(sample_size / 2)
                center_j = j + int(sample_size / 2)
                half_output = int(output_size / 2)
                mask_crop = self.convert_to_onehot(
                    binary_mask[center_i - half_output: center_i + half_output + 1,
                    center_j - half_output:center_j + half_output + 1])
                contour_mask_crop = self.convert_to_onehot(
                    contour_mask[center_i - half_output: center_i + half_output + 1,
                    center_j - half_output:center_j + half_output + 1])
                if train:
                    self.train_data.append(image_crop)
                    self.train_mask.append(mask_crop)
                    self.train_contour_mask.append(contour_mask_crop)
                else:
                    self.valid_data.append(image_crop)
                    self.valid_mask.append(mask_crop)
                    self.valid_contour_mask.append(contour_mask_crop)

                np.save(save_folder + filename + str(i) + str(j) + "_image", image_crop)
                np.save(save_folder + filename + str(i) + str(j) + "_mask", mask_crop)
                np.save(save_folder + filename + str(i) + str(j) + "_contour", contour_mask_crop)
        # for i in range(sample_number):
        #     x = random.randint(0, width - sample_size)
        #     y = random.randint(0, height - sample_size)
        #     image_crop = image[x:x + sample_size, y:y + sample_size, 0:3]
        #     mask_crop = self.convert_to_onehot(binary_mask[x:x + sample_size, y:y + sample_size])
        #     contour_mask_crop = self.convert_to_onehot(contour_mask[x:x + sample_size, y:y + sample_size])
        #     if train:
        #         self.train_data.append(image_crop)
        #         self.train_mask.append(mask_crop)
        #         self.train_contour_mask.append(contour_mask_crop)
        #     else:
        #         self.valid_data.append(image_crop)
        #         self.valid_mask.append(mask_crop)
        #         self.valid_contour_mask.append(contour_mask_crop)
        #     # save image as np
        #     np.save(save_folder + filename + str(i) + "_image", image_crop)
        #     np.save(save_folder + filename + str(i) + "_mask", mask_crop)
        #     np.save(save_folder + filename + str(i) + "_contour", contour_mask_crop)

    def verification_data(self):
        sampled_valid_data = []
        sampled_valid_mask = []
        sampled_valid_contour = []
        for i in range(0, self.valid_size):
            sampled_valid_data.append(self.valid_data[i])
            sampled_valid_mask.append(self.valid_mask[i])
            sampled_valid_contour.append(self.valid_contour_mask[i])
        return sampled_valid_data, sampled_valid_mask, sampled_valid_contour

    def test_data(self):
        test_data = []
        test_mask = []
        for filename in os.listdir(self.test_dir):
            if filename.endswith(".png"):
                filename_suffix = filename.split(".")[0]
                test_data.append(mc.imread(self.test_dir + filename))
                test_mask.append(np.load(self.test_dir + filename_suffix + ".npy"))
        return test_data, test_mask

    def get_train_data(self):
        train_data = []
        train_mask = []
        for filename in os.listdir(self.train_dir):
            if filename.endswith(".png"):
                filename_suffix = filename.split(".")[0]
                train_data.append(mc.imread(self.train_dir + filename))
                train_mask.append(np.load(self.train_dir + filename_suffix + ".npy"))
        return train_data, train_mask

    def convert_to_onehot(self, array):
        result = np.zeros((array.shape[0], array.shape[1], self.num_class), dtype=np.float32)
        for i in range(self.num_class):
            layer = np.zeros(array.shape)
            layer[np.where(array == i)] = 1
            result[..., i] = layer
        return result


def find_contour(add_mask):
    """
    find the contour give segmentation mask
    :param add_mask: the segmentation mask
    :return: contour mask
    """
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
    train_dir = "/home/yunzhe/norm_data/training_data/"
    test_dir = "/home/yunzhe/norm_data/test_data/"

    train_number = 25
    file_count = 0
    filter = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    for filename in os.listdir(data_dir):
        file_count += 1
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
            contour_mask = np.zeros((width, height), np.uint8)
            distance_mask = np.zeros((width, height), np.float32)
            # add_mask = np.zeros((width, height))

            if os.path.exists(train_dir + filename_suffix + "_distance.npy"):
                continue
            # extract each nuclear
            region_label = 1
            for region in root.iter("Region"):
                polygon = []
                for vertex in region.iter("Vertex"):
                    polygon.append((float(vertex.attrib['X']), float(vertex.attrib['Y'])))
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                path = Path(polygon)
                grid = path.contains_points(points)
                grid = grid.reshape((width, height)).astype(np.int32)
                mask[np.where(grid == 1)] = region_label
                region_label += 1
                grid_dilation = dilation(grid, filter)
                grid_erosion = erosion(grid, filter)
                # calculate distance map
                contour = grid_dilation - grid_erosion
                contour_points = np.where(contour == 1)
                nuclei_points = np.where(grid == 1)
                sub_dist_map = np.zeros((width, height))
                for i in range(len(nuclei_points[0])):
                    x = nuclei_points[0][i]
                    y = nuclei_points[1][i]
                    if contour[x][y] == 1:
                        continue
                    min_dist = 10000
                    for j in range(len(contour_points[0])):
                        contour_x = contour_points[0][j]
                        contour_y = contour_points[1][j]
                        dist = math.sqrt((x - contour_x) ** 2 + (y - contour_y) ** 2)
                        min_dist = min(dist, min_dist)
                    sub_dist_map[x][y] = min_dist
                if np.max(sub_dist_map) > 0:
                    sub_dist_map = sub_dist_map / np.max(sub_dist_map)
                contour_mask = np.logical_or(contour_mask, (grid_dilation - grid_erosion).astype(np.uint8))
                distance_mask[nuclei_points] = 0
                distance_mask += sub_dist_map
                # cv2.drawContours(contour_mask, [np.asarray(polygon).astype(int)], 0, 1, 3)

            # mask[np.where(contour_mask == 1)] = 0
            plt.imshow(distance_mask)
            plt.show()
            train_set = set(os.listdir(train_dir))
            # save the result for training
            if (filename_suffix + ".png") in train_set:
                # image.save(train_dir + filename_suffix + ".png")
                np.save(train_dir + filename_suffix, mask)
                np.save(train_dir + filename_suffix + "_contour", contour_mask)
                np.save(train_dir + filename_suffix + "_distance", distance_mask)
            else:
                # image.save(test_dir + filename_suffix + ".png")
                np.save(test_dir + filename_suffix, mask)
                np.save(test_dir + filename_suffix + "_contour", contour_mask)
                np.save(test_dir + filename_suffix + "_distance", distance_mask)
            plt.imshow(contour_mask)
            plt.show()
            plt.imshow(mask)
            plt.show()
