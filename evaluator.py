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

    def evaluate(self, mode=1):
        test_data, test_mask = self.data_provider.test_data()
        # test_data, test_mask = self.data_provider.get_train_data()
        self.net.load_model(self.model_path)
        average_f1 = 0.0
        average_aji = 0.0
        average_dice = 0.0
        for i in range(len(test_data)):
            if mode == 1:
                mask = self.predict(test_data[i])
            else:
                mask = self.predict_with_sliding_window(test_data[i])
            plt.imshow(mask)
            plt.show()

            plt.imshow(test_mask[i])
            plt.show()

            plt.imshow(test_data[i])
            plt.show()

            f1_score, miss_map, false_map = self.f1_score(test_mask[i].astype(np.int32), mask.astype(np.int32))
            average_f1 += f1_score
            aji = self.aji(test_mask[i].astype(np.int32), mask.astype(np.int32))
            average_aji += aji
            average_dice += self.dice(test_mask[i].astype(np.int32), mask.astype(np.int32))
            plt.imshow(miss_map)
            plt.show()
            plt.imshow(false_map)
            plt.show()

        print("average f1 " + str(average_f1 / len(test_data)))
        print("average aji " + str(average_aji / len(test_data)))
        print("average dice " + str(average_dice / len(test_data)))

    @staticmethod
    def f1_score(true_mask, predict_mask, threshold=0.5):
        missed_map = np.zeros(true_mask.shape)
        false_map = np.zeros(true_mask.shape)
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
            else:
                missed_map[np.where(true_mask == target_region.label)] = 1
        for left_region in predict_regions:
            false_map[np.where(predict_mask == left_region.label)] = 1
        fn = len(ground_truth) - tp
        fp = len(predict_regions)
        f1 = float(2 * tp) / (2 * tp + fp + fn)
        print("fp " + str(fp))
        print("fn " + str(fn))
        print("tp " + str(tp))
        print("f1 score " + str(f1))
        return f1, missed_map, false_map

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

    def predict(self, test_image, output_size=219):
        """
        given an image, predict segmentation mask
        :param test_image: image array
        :return: mask array
        """
        test_image = (test_image - np.average(test_image)) / np.std(test_image)

        height, width = test_image.shape[0], test_image.shape[1]
        mask = np.zeros((height, width))
        contour = np.zeros((height, width))
        test_image = np.pad(test_image, ((self.sample_size, self.sample_size), (self.sample_size, self.sample_size),
                            (0, 0)), "reflect")
        x, y = 0, 0
        stride = int((self.sample_size - output_size) / 2)
        while x < height or y < width:
            sample = test_image[x + self.sample_size - stride: x + self.sample_size + output_size + stride,
                     y + self.sample_size - stride: y + self.sample_size + output_size + stride, 0:3]
            x_max = min(x + output_size, height)
            y_max = min(y + output_size, width)
            # sample = np.zeros((self.sample_size, self.sample_size, 3)).astype(np.int32)
            # sample[:x_max - x, :y_max - y, :] = test_image[x:x_max, y:y_max, 0:3]
            # # mirror the sample
            # if x_max - x < self.sample_size:
            #     sample[x_max - x: self.sample_size, :y_max - y, :] = np.flip(test_image[x_max - x - self.sample_size:,
            #                                                                  y:y_max, 0:3], axis=0)
            # if y_max - y < self.sample_size:
            #     sample[:x_max - x, y_max - y: self.sample_size, :] = np.flip(
            #         test_image[x: x_max, y_max - y - self.sample_size:,
            #         0:3], axis=1)
            # if x_max - x < self.sample_size and y_max - y < self.sample_size:
            #     sample[x_max - x: self.sample_size, y_max - y: self.sample_size, :] = np.flip(
            #         np.flip(test_image[x_max - x - self.sample_size:,
            #                 y_max - y - self.sample_size:, 0:3],
            #                 axis=1), axis=0)
            sample_list = [sample, (np.rot90(sample, 2)), (np.flip(sample, axis=0)), (np.flip(sample, axis=1))]
            sample_mask_list, sample_contour_list = self.net.predict(np.asarray(sample_list))
            sample_mask = np.average(np.asarray([sample_mask_list[0], np.rot90(sample_mask_list[1], 2),
                                                 np.flip(sample_mask_list[2], axis=0),
                                                 np.flip(sample_mask_list[3], axis=1)]), axis=0)
            sample_contour = np.average(np.asarray([sample_contour_list[0], np.rot90(sample_contour_list[1], 2),
                                                    np.flip(sample_contour_list[2], axis=0),
                                                    np.flip(sample_contour_list[3], axis=1)]), axis=0)
            mask[x:x_max, y:y_max] = sample_mask[:x_max - x, :y_max - y]
            contour[x:x_max, y:y_max] = sample_contour[:x_max - x, :y_max - y]

            # update index
            if x + output_size < height:
                x += output_size
            elif y + output_size < width:
                y += output_size
                x = 0
            else:
                break
        return self.post_process(mask, contour)

    def predict_with_sliding_window(self, test_image, normalization=True):
        """
        given a test image, generate final segmentation result
        :param test_image: test image
        :return: segmentation result
        """
        if normalization:
            test_image = (test_image - np.average(test_image)).astype(np.float32) / np.std(test_image)
            # test_image = test_image / 255.
            # test_image = (test_image - np.average(test_image)).astype(np.float32)
        width, height, _ = test_image.shape
        nuclei_map = np.zeros((width, height), dtype=np.float32)
        contour_map = np.zeros((width, height), dtype=np.float32)
        half_size = int((self.sample_size - 1) / 2)
        test_image = np.pad(test_image, ((half_size, half_size),(half_size, half_size), (0, 0)), "reflect")
        for i in range(width):
            input_sample = np.zeros((height, self.sample_size, self.sample_size, 3))
            for j in range(height):
                input_sample[j,...] = test_image[i: i + self.sample_size, j: j + self.sample_size, :]
            probs = self.net.predict(input_sample)
            for j in range(height):
                input_sample[j, ...] = np.rot90(test_image[i: i + self.sample_size, j: j + self.sample_size, :], 1,
                                                (0, 1))
            rot_probs = self.net.predict(input_sample)
            for j in range(height):
                input_sample[j, ...] = np.rot90(test_image[i: i + self.sample_size, j: j + self.sample_size, :], 2,
                                                (0, 1))
            rot1_probs = self.net.predict(input_sample)
            for j in range(height):
                input_sample[j, ...] = np.rot90(test_image[i: i + self.sample_size, j: j + self.sample_size, :], 3,
                                                (0, 1))
            rot2_probs = self.net.predict(input_sample)
            for j in range(height):
                input_sample[j, ...] = np.flip(test_image[i: i + self.sample_size, j: j + self.sample_size, :], 0)
            flip_probs = self.net.predict(input_sample)
            for j in range(height):
                input_sample[j, ...] = np.flip(test_image[i: i + self.sample_size, j: j + self.sample_size, :], 1)
            flip1_probs = self.net.predict(input_sample)
            for j in range(height):
                nuclei_map[i][j] = (probs[j][1] + rot_probs[j][1] + rot1_probs[j][1] + rot2_probs[j][1] +
                                    flip_probs[j][1] + flip1_probs[j][1]) / 6
                contour_map[i][j] = (probs[j][2] + rot_probs[j][2] + rot1_probs[j][2] +
                                     rot2_probs[j][2] + flip_probs[j][2] + flip1_probs[j][2]) / 6
        # post process
        plt.imshow(nuclei_map)
        plt.show()
        result_map = self.post_process(nuclei_map, contour_map)
        plt.show()
        plt.imshow(contour_map)
        plt.show()
        plt.imshow(result_map)
        plt.show()
        return result_map

    def post_process(self, nuclei_map, contour_map, threshold=0.5):
        plt.imshow(nuclei_map)
        plt.show()
        plt.imshow(contour_map)
        plt.show()
        average_contour = np.sum(contour_map) / len(np.where(contour_map > 0)[0])
        nuclei_map = nuclei_map - contour_map
        # nuclei_map[np.where(contour_map > 0.5)] = 0
        nuclei_map[nuclei_map > threshold] = 1
        nuclei_map[nuclei_map <= threshold] = 0
        result_map = np.zeros(nuclei_map.shape)
        label_map = measure.label(nuclei_map)
        region_props = measure.regionprops(label_map)
        kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        for region in region_props:
            temp_map = np.zeros(nuclei_map.shape)
            if region.area > 50:
                temp_map[np.where(label_map == region.label)] = 1
                for i in range(2):
                    temp_contour = np.multiply(temp_map, contour_map)
                    average_temp = np.sum(temp_contour) / len(np.where(temp_contour > 0)[0])
                    if average_temp > average_contour:
                        temp_map = dilation(temp_map, square(3))
                        break
                    temp_map = dilation(temp_map, square(3))
                result_map[np.where(temp_map == 1)] = region.label
        return result_map

    @staticmethod
    def local_maximum(contour_map, x, y):
        heigth, width = contour_map.shape
        if x > 0 and contour_map[x][y] > contour_map[x - 1][y] and \
            y > 0 and contour_map[x][y] > contour_map[x][y - 1] and \
            x < heigth - 1 and contour_map[x][y] > contour_map[x + 1][y] \
            and y < width - 1 and contour_map[x][y] > contour_map[x][y + 1]:
            return True
        return False

    @staticmethod
    def get_segment_object(mask, contour, area_thresh=30):
        """
        get the final segmentation result based on mask and contour
        :param mask: mask result
        :param contour: contour result
        :param area_thresh: area threshold for minimum nuclei area
        :return: final result, with unique integer for one region
        """

        # mask[np.where(contour == 1)] = 0
        # mask = dilation(mask, square(5))
        mask = mask - contour
        mask[np.where(mask > 0.5)] = 1
        label_mask = measure.label(mask)
        region_props = measure.regionprops(label_mask)

        # remove small area
        for region in region_props:
            if region.area < area_thresh:
                points = region.coords
                for point in points:
                    label_mask[point[0]][point[1]] = 0
        # label_mask = dilation(label_mask, square(3))
        return label_mask


if __name__ == "__main__":
    from net import deep_contour_net, classification_net
    from data import segmentation_provider, patch_provider

    net = deep_contour_net.DeepContourNet(cost="dice", sample_size=225)
    data_provider = segmentation_provider.SegmentationDataProvider("/data/Cell/norm_data/training_data/",
                                                                   "/data/Cell/norm_data/test_data/", sample_size=225,
                                                                   test=True)
    # net = classification_net.SimpleNet(sample_size=51)
    # data_provider = patch_provider.PatchProvider("/data/Cell/norm_data/training_data/",
    #                                              "/data/Cell/norm_data/test_data/", test=True)
    evaluator = Evaluator("/data/Cell/yunzhe/cross_entropy/model.ckpt1",  net, data_provider, sample_size=225)
    evaluator.evaluate(mode=1)
