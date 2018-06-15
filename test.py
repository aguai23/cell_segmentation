from net.deep_contour_net import DeepContourNet
from data.segmentation_provider import SegmentationDataProvider
from matplotlib import pyplot as plt
import numpy as np
data_provider = SegmentationDataProvider("/home/cell/training_data/")
net = DeepContourNet()
test_data, test_mask, test_contour = data_provider.verification_data()
# for i in range(5):
#     plt.imshow(test_data[i])
#     plt.show()
#     input = np.reshape(test_data[i], [1] + list(test_data[i].shape))
#     mask, contour = net.predict("/home/cell/yunzhe/miccai_code/model/model.ckpt200", input)
#     plt.imshow(mask, cmap="gray")
#     plt.show()

for i in range(5):
    train_data, train_mask, _ = data_provider(1)
    mask, contour = net.predict("/home/cell/yunzhe/miccai_code/model/model.ckpt300", train_data)
    plt.imshow(mask, cmap="gray")
    plt.show()
    # plt.imshow(contour, cmap="gray")
    # plt.show()