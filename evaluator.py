import logging
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

class Evaluator(object):

    def __init__(self, model_path, net, data_provider, sample_size=224):
        self.model_path = model_path
        self.net = net
        self.data_provider = data_provider
        self.sample_size = sample_size

    def f1_score(self):
        test_data, test_mask = self.data_provider.test_data()
        self.net.load_model(self.model_path)
        for i in range(5):
            plt.imshow(test_data[i])
            plt.show()
            plt.imshow(test_mask[i])
            plt.show()
            mask = self.predict(test_data[i])
            print(np.sum(mask))
            plt.imshow(mask)
            plt.show()

    def predict(self, test_image):
        height, width = test_image.shape[0], test_image.shape[1]
        mask = np.zeros((height, width))
        contour = np.zeros((height, width))
        x, y = 0, 0
        self.net.load_model(self.model_path)
        while x < height or y < width:
            x_max = min(x + self.sample_size, height)
            y_max = min(y + self.sample_size, width)
            sample = np.zeros((self.sample_size, self.sample_size, 3)).astype(np.int32)
            sample[:x_max - x, :y_max - y, :] = test_image[x:x_max, y:y_max, 0:3]
            sample = np.reshape(sample, [1] + list(sample.shape))

            sample_mask, sample_contour = self.net.predict(sample)
            mask[x:x_max,y:y_max] = sample_mask[:x_max - x, :y_max - y]
            contour[x:x_max, y:y_max] = sample_contour[:x_max - x, :y_max - y]

            # update index
            if x + self.sample_size < height:
                x += self.sample_size
            elif y + self.sample_size < width:
                y += self.sample_size
                x = 0
            else:
                break
        return self.get_segment_object(mask, contour)

    def get_segment_object(self, mask, contour, area_thresh=5):
        """
        get the final segmentation result based on mask and contour
        :param mask: mask result
        :param contour: contour result
        :param area_thresh: area threshold for minimum nuclei area
        :return: final result, with unique integer for one region
        """

        mask[np.where(contour == 1)] = 0
        label_mask = measure.label(mask)
        region_props = measure.regionprops(label_mask)

        for region in region_props:
            if region.area < area_thresh:
                points = region.coords
                for point in points:
                    label_mask[point[0]][point[1]] = 0
        return label_mask


if __name__ == "__main__":
    from net import deep_contour_net
    from data import segmentation_provider
    net = deep_contour_net.DeepContourNet()
    data_provider = segmentation_provider.SegmentationDataProvider("/home/cell/training_data/training_data/",
                                                                   "/home/cell/training_data/test_data/")
    evaluator = Evaluator("/home/cell/yunzhe/miccai_code/model/model.ckpt100",net, data_provider)
    evaluator.f1_score()
