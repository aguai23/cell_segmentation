from data import segmentation_provider, patch_provider
from trainer import Trainer
from net.deep_contour_net import DeepContourNet
from net.classification_net import SimpleNet

# data_provider = segmentation_provider.SegmentationDataProvider("/home/cell/norm_data/training_data/",
#                                                                "/home/cell/norm_data/test_data/")
# deepNet = DeepContourNet()
# trainer = Trainer(deepNet, batch_size=1, optimizer="adam", learning_rate=0.0001, decay_rate=0.9, decay_step=5000)
# trainer.train(data_provider, "/home/cell/yunzhe/miccai_code/model/", training_iters=64, epochs=501,
#               save_epoch=50, restore=True, verify_epoch=50, display_step=5)

data_provider = patch_provider.PatchProvider("/home/cell/norm_data/training_data/",
                                             "/home/cell/norm_data/test_data/", data_augmentation=False)

simpleNet = SimpleNet()
trainer = Trainer(simpleNet, batch_size=128, optimizer="momentum",
                  learning_rate=0.01, decay_rate=0.9, decay_step=10000, momentum=0.5)
trainer.train_classification(data_provider, "/home/cell/yunzhe/miccai_code/classify_model", training_iters=1000, epochs=101,
                             save_epoch=5, restore=True, verify_epoch=5, display_step=100)