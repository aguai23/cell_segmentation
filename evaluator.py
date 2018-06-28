import logging
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from skimage.morphology import square, dilation, erosion


class Evaluator(object):

    def __init__(self, model_path, net, data_provider, sample_size=224):
        self.model_path = model_path
        self.net = net
        self.data_provider = data_provider
        self.sample_size = sample_size

    def evaluate(self):
        test_data, test_mask = self.data_provider.test_data()
        self.net.load_model(self.model_path)
        average_f1 = 0.0
        average_aji = 0.0
        average_dice = 0.0
        for i in range(len(test_data)):
            mask = self.predict(test_data[i])
            plt.imshow(test_mask[i])
            plt.show()
            plt.imshow(mask)
            plt.show()
            f1_score = self.f1_score(test_mask[i].astype(np.int32), mask.astype(np.int32))
            average_f1 += f1_score
            aji = self.aji(test_mask[i].astype(np.int32), mask.astype(np.int32))
            average_aji += aji
            average_dice += self.dice(test_mask[i].astype(np.int32), mask.astype(np.int32))

        print("average f1 " + str(average_f1 / len(test_data)))
        print("average aji " + str(average_aji / len(test_data)))
        print("average dice " + str(average_dice / len(test_data)))

    @staticmethod
    def f1_score(true_mask, predict_mask, threshold=0.5):
        ground_truth = measure.regionprops(true_mask)
        predict_regions = measure.regionprops(predict_mask)
        tp = 0
        for target_region in ground_truth:
            target_box = target_region.bbox
            target_set = set()
            region_to_remove = None
            for coord in target_region.coords:
                target_set.add(str(coord))
            for predict_region in predict_regions:
                predict_box = predict_region.bbox
                if target_box[2] > predict_box[0] and predict_box[2] > target_box[0] \
                        and target_box[3] > predict_box[1] and predict_box[3] > target_box[1]:
                    # overlap, calculate iou
                    predict_set = set()
                    for coord in predict_region.coords:
                        predict_set.add(str(coord))
                    union = len(target_set & predict_set)
                    iou = float(union) / (target_region.area + predict_region.area - union)
                    if iou > threshold:
                        tp += 1
                        region_to_remove = predict_region
                        break
            if region_to_remove:
                predict_regions.remove(region_to_remove)
        fn = len(ground_truth) - tp
        fp = len(predict_regions)
        f1 = float(2 * tp) / (2 * tp + fp + fn)
        print("fp " + str(fp))
        print("fn " + str(fn))
        print("tp " + str(tp))
        print("f1 score " + str(f1))
        return f1

    @staticmethod
    def aji(true_mask, predict_mask):
        ground_truth = measure.regionprops(true_mask)
        predict_regions = measure.regionprops(predict_mask)
        c = 0.0
        u = 0.0
        for target_region in ground_truth:
            target_box = target_region.bbox
            target_set = set()
            region_to_remove = None
            max_iou = 0.0
            sub_c = 0.0
            sub_u = 0.0
            for coord in target_region.coords:
                target_set.add(str(coord))
            for predict_region in predict_regions:
                predict_box = predict_region.bbox
                if target_box[2] > predict_box[0] and predict_box[2] > target_box[0] \
                        and target_box[3] > predict_box[1] and predict_box[3] > target_box[1]:
                    # overlap, calculate iou
                    predict_set = set()
                    for coord in predict_region.coords:
                        predict_set.add(str(coord))
                    union = len(target_set & predict_set)
                    iou = float(union) / (target_region.area + predict_region.area - union)
                    if iou > max_iou:
                        max_iou = iou
                        region_to_remove = predict_region
                        sub_c = union
                        sub_u = target_region.area + predict_region.area - union
            if max_iou == 0:
                u += target_region.area
            else:
                u += sub_u
                c += sub_c
            if region_to_remove:
                predict_regions.remove(region_to_remove)

        for left_region in predict_regions:
            u += left_region.area
        aji = c / u
        print("aji " + str(aji))
        return aji

    @staticmethod
    def dice(true_mask, predict_mask):
        height, width = true_mask.shape
        overlap = 0
        union = 0
        for i in range(height):
            for j in range(width):
                if true_mask[i][j] > 0 and predict_mask[i][j] > 0:
                    overlap += 1
                    union += 1
                elif true_mask[i][j] > 0 or predict_mask[i][j] > 0:
                    union += 1
        dice = float(overlap) / union
        print("dice " + str(dice))
        return dice

    def predict(self, test_image):
        """
        given an image, predict segmentation mask
        :param test_image: image array
        :return: mask array
        """
        height, width = test_image.shape[0], test_image.shape[1]
        mask = np.zeros((height, width))
        contour = np.zeros((height, width))
        x, y = 0, 0
        while x < height or y < width:
            x_max = min(x + self.sample_size, height)
            y_max = min(y + self.sample_size, width)
            sample = np.zeros((self.sample_size, self.sample_size, 3)).astype(np.int32)
            sample[:x_max - x, :y_max - y, :] = test_image[x:x_max, y:y_max, 0:3]
            # mirror the sample
            if x_max - x < self.sample_size:
                sample[x_max - x: self.sample_size, :y_max - y, :] = np.flip(test_image[x_max - x - self.sample_size:,
                                                                             y:y_max, 0:3], axis=0)
            if y_max - y < self.sample_size:
                sample[:x_max - x, y_max - y: self.sample_size, :] = np.flip(
                    test_image[x: x_max, y_max - y - self.sample_size:,
                    0:3], axis=1)
            if x_max - x < self.sample_size and y_max - y < self.sample_size:
                sample[x_max - x: self.sample_size, y_max - y: self.sample_size, :] = np.flip(
                    np.flip(test_image[x_max - x - self.sample_size:,
                            y_max - y - self.sample_size:, 0:3],
                            axis=1), axis=0)
            sample = np.reshape(sample, [1] + list(sample.shape))
            sample_mask, sample_contour = self.net.predict(sample)
            mask[x:x_max, y:y_max] = sample_mask[:x_max - x, :y_max - y]
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

    @staticmethod
    def get_segment_object(mask, contour, area_thresh=50):
        """
        get the final segmentation result based on mask and contour
        :param mask: mask result
        :param contour: contour result
        :param area_thresh: area threshold for minimum nuclei area
        :return: final result, with unique integer for one region
        """

        mask[np.where(contour == 1)] = 0
        mask = dilation(mask, square(5))
        label_mask = measure.label(mask)
        region_props = measure.regionprops(label_mask)

        # remove small area
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
    evaluator = Evaluator("/home/cell/yunzhe/miccai_code/model/model.ckpt30", net, data_provider)
    evaluator.evaluate()
