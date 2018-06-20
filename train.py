from data import segmentation_provider
from trainer import Trainer
from net.deep_contour_net import DeepContourNet

data_provider = segmentation_provider.SegmentationDataProvider("/home/cell/training_data/")
deepNet = DeepContourNet()
trainer = Trainer(deepNet, batch_size=1, optimizer="adam", learning_rate=0.0001, decay_rate=0.9, decay_step=100)
trainer.train(data_provider, "/home/cell/yunzhe/miccai_code/model/", training_iters=64, epochs=101,
              save_epoch=10, restore=False, verify_epoch=5, display_step=5)