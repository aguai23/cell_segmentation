import os
from shutil import rmtree
import scipy.misc as mc
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class PatchProvider(object):

    def __init__(self, train_dir, test_dir, train_percent=0.7, sample_size=51, sample_number=5000,
                 num_class=3, shuffle_data=True,
                 resample=False, data_augmentation=False):
        # array to store the data
        self.train_data = []
        self.valid_data = []
        self.train_label = []
        self.valid_label = []

        self.index = -1
        self.sample_size = sample_size
        self.num_class = num_class
        self.test_dir = test_dir
        train_folder = train_dir + "train/"
        valid_folder = train_dir + "valid/"

        if resample or (not os.path.exists(train_folder)):
            # remove previous data
            if os.path.exists(train_folder):
                rmtree(train_folder)
                rmtree(valid_folder)

            # resample the training and valid data
            filenames = []
            # read data from file
            for filename in os.listdir(train_dir):
                if filename.endswith(".png"):
                    filenames.append(filename)

            self.train_size = int(train_percent * len(filenames))
            self.valid_size = len(filenames) - self.train_size

            if not os.path.exists(train_folder):
                os.mkdir(train_folder)
            if not os.path.exists(valid_folder):
                os.mkdir(valid_folder)

            count = 0
            for filename in filenames:
                filename_suffix = filename.split(".")[0]
                image = mc.imread(train_dir + filename)
                mask = np.load(train_dir + filename_suffix + ".npy")
                contour_mask = np.load(train_dir + filename_suffix + "_contour.npy")

                if count < self.train_size:
                    self.sample_data(filename, image, mask,
                                     contour_mask, True, sample_size, sample_number, train_folder)
                elif count:
                    self.sample_data(filename, image, mask,
                                     contour_mask, False, sample_size, sample_number, valid_folder)
                count += 1
            self.train_size = len(self.train_data)
            self.valid_size = len(self.valid_data)
        else:
            # load saved data
            for filename in os.listdir(train_folder):
                if filename.endswith("_contour.npy"):
                    self.train_data.append(np.load(train_folder + filename))
                    self.train_label.append(self.convert_to_onehot(2))
                elif filename.endswith("_nuclei.npy"):
                    self.train_data.append(np.load(train_folder + filename))
                    self.train_label.append(self.convert_to_onehot(1))
                else:
                    self.train_data.append(np.load(train_folder + filename))
                    self.train_label.append(self.convert_to_onehot(0))

            for filename in os.listdir(valid_folder):
                if filename.endswith("_contour.npy"):
                    self.valid_data.append(np.load(valid_folder + filename))
                    self.valid_label.append(self.convert_to_onehot(2))
                elif filename.endswith("_nuclei.npy"):
                    self.valid_data.append(np.load(valid_folder + filename))
                    self.valid_label.append(self.convert_to_onehot(1))
                else:
                    self.valid_data.append(np.load(valid_folder + filename))
                    self.valid_label.append(self.convert_to_onehot(0))

            self.train_size = len(self.train_data)
            self.valid_size = len(self.valid_data)
        print(self.train_size)
        print(self.valid_size)

        if data_augmentation:
            augment_train_data = []
            augment_train_label = []
            for i in range(self.train_size):
                augment_train_data.append(self.train_data[i])
                augment_train_label.append(self.train_label[i])

                augment_train_data.append(np.rot90(self.train_data[i], 1, (0,1)))
                augment_train_label.append(self.train_label[i])

                augment_train_data.append(np.rot90(self.train_data[i], 2, (0, 1)))
                augment_train_label.append(self.train_label[i])

                augment_train_data.append(np.rot90(self.train_data[i], 3, (0, 1)))
                augment_train_label.append(self.train_label[i])

                augment_train_data.append(np.flip(self.train_data[i], 0))
                augment_train_label.append(self.train_label[i])

                augment_train_data.append(np.flip(self.train_data[i], 1))
                augment_train_label.append(self.train_label[i])

            self.train_data = augment_train_data
            self.train_label = augment_train_label

        print("augmented train data " + str(len(self.train_data)))
        if shuffle_data:
            shuffled_train_data, shuffled_train_label = shuffle(self.train_data, self.train_label)
            self.train_data = shuffled_train_data
            self.train_label = shuffled_train_label

    def __call__(self, size):
        X = np.zeros((size, self.sample_size, self.sample_size, 3))
        Y = np.zeros((size, self.num_class))

        for i in range(size):
            self.index += 1
            if self.index >= self.train_size:
                self.index = 0
            X[i] = self.train_data[self.index]
            Y[i] = self.train_label[self.index]
        return np.asarray(X).astype(np.float32), np.asarray(Y).astype(np.float32)

    def verification_data(self):
        return np.asarray(self.valid_data), np.asarray(self.valid_label)

    def test_data(self):
        test_data = []
        test_mask = []
        for filename in os.listdir(self.test_dir):
            if filename.endswith(".png"):
                filename_suffix = filename.split(".")[0]
                test_data.append(mc.imread(self.test_dir + filename))
                test_mask.append(np.load(self.test_dir + filename_suffix + ".npy"))
        return test_data, test_mask

    def convert_to_onehot(self, label):
        onehot = np.zeros(self.num_class)
        onehot[label] = 1
        return onehot

    def sample_data(self, filename, image, mask, contour_mask, train, sample_size, sample_number, save_folder):
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
        heigth, width, _ = image.shape
        half_size = int((sample_size - 1) / 2)
        contour_points = np.where(contour_mask == 1)
        nuclei_points = np.where(mask > 0)
        background_points = np.where(mask == 0)

        # sample contour patch
        index = 0
        count = 0
        while count < sample_number and index < len(contour_points[0]):
            # sample contour patch
            x = contour_points[0][index]
            y = contour_points[1][index]

            assert contour_mask[x][y] == 1
            if x - half_size < 0 or y - half_size < 0 or x + half_size + 1 > heigth or y + half_size + 1 > width:
                index += 1
                continue
            if train:
                self.train_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.train_label.append(self.convert_to_onehot(2))
            else:
                self.valid_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.valid_label.append(self.convert_to_onehot(2))
            count += 1
            index += 1
            np.save(save_folder + filename + str(index) + "_contour",
                    image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])

        # sample nuclei patch
        index = 0
        count = 0
        while count < sample_number and index < len(nuclei_points[0]):
            x = nuclei_points[0][index]
            y = nuclei_points[1][index]

            assert mask[x][y] > 0
            if x - half_size < 0 or y - half_size < 0 or x + half_size + 1 > heigth or y + half_size + 1 > width or \
                    contour_mask[x][y] == 1:
                index += 1
                continue
            if train:
                self.train_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.train_label.append(self.convert_to_onehot(1))
            else:
                self.valid_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.valid_label.append(self.convert_to_onehot(1))
            count += 1
            index += 1
            np.save(save_folder + filename + str(index) + "_nuclei",
                    image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])

        # sample background patch
        index = 0
        count = 0
        while count < sample_number and index < len(background_points[0]):
            x = background_points[0][index]
            y = background_points[1][index]

            assert mask[x][y] == 0
            if x - half_size < 0 or y - half_size < 0 or x + half_size + 1 > heigth or y + half_size + 1 > width or \
                    contour_mask[x][y] == 1:
                index += 1
                continue
            if train:
                self.train_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.train_label.append(self.convert_to_onehot(0))
            else:
                self.valid_data.append(image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])
                self.valid_label.append(self.convert_to_onehot(0))
            count += 1
            index += 1
            np.save(save_folder + filename + str(index) + "_background",
                    image[x - half_size: x + half_size + 1, y - half_size: y + half_size + 1, 0:3])


if __name__ == "__main__":
    data_provider = PatchProvider("/home/cell/norm_data/training_data/",
                                  "/home/cell/norm_data/test_data/")
    train_data, train_label = data_provider(5)
    for i in range(5):
        plt.imshow(train_data[i].astype(np.int32))
        plt.show()
        print(train_label[i])
