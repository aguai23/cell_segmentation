from net.deep_contour_net import DeepContourNet
from data.segmentation_provider import SegmentationDataProvider
from data.patch_provider import PatchProvider
from net.classification_net import SimpleNet
from matplotlib import pyplot as plt
import numpy as np

data_provider = PatchProvider("/data/Cell/norm_data/training_data/",
                              "/data/Cell/norm_data/test_data/")

simpleNet = SimpleNet()
test_data, test_label = data_provider.verification_data()
for i in range(50000, 50010):
    simpleNet.load_model("/home/cell/yunzhe/miccai_code/classify_model/model.ckpt5")
    logits = simpleNet.predict(test_data[i:i + 1,...], test_label[i:i+1, ...])
    print(test_label[i])
    print(logits)
# data_provider = SegmentationDataProvider("/home/cell/training_data/training_data/",
#                                                                "/home/cell/training_data/test_data/")
# net = DeepContourNet()
# test_data, test_mask, test_contour = data_provider.verification_data()
# for i in range(100,105):
#     plt.imshow(test_data[i])
#     plt.show()
#     plt.imshow(np.argmax(test_mask[i], axis=2))
#     plt.show()
#     input = np.reshape(test_data[i], [1] + list(test_data[i].shape))
#     mask_label = np.reshape(test_mask[i],[1] + list(test_mask[i].shape))
#     mask_contour = np.reshape(test_contour[i],[1] + list(test_contour[i].shape))
#     net.load_model("/home/cell/yunzhe/miccai_code/model/model.ckpt100")
#     mask1, contour1 = net.predict(input, mask_label, mask_contour)
#
#     plt.imshow(mask1, cmap="gray")
#     plt.show()

# for i in range(5):
#     train_data, train_mask, train_contour = data_provider(1)
#     # plt.imshow(train_data[0].astype(np.int32))
#     # plt.show()
#     true_mask = np.argmax(train_mask[0], axis=2)
#     true_contour = np.argmax(train_contour[0], axis=2)
#     # plt.imshow(true_mask, cmap="gray")
#     # plt.show()
#     plt.imshow(true_contour, cmap="gray")
#     plt.show()
#     mask, contour = net.predict("/home/cell/yunzhe/miccai_code/model/model.ckpt100", train_data,
#                                 train_mask, train_contour)
#     plt.imshow(contour, cmap="gray")
#     plt.show()

# plt.imshow(contour, cmap="gray")
# plt.show()
