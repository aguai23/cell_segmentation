from data import segmentation_provider
from trainer.trainer import Trainer
from net.deep_contour_net import DeepContourNet

deepNet = DeepContourNet(cost="fuse", sample_size=225, output_size=225)
data_provider = segmentation_provider.SegmentationDataProvider("/data/Cell/norm_data_new/",
                                                               "/data/Cell/norm_data/test_data/", resample=True,
                                                               sample_number=3000, sample_size=225, output_size=225,
                                                               train_percent=0.9)

trainer = Trainer(deepNet, batch_size=1, optimizer="adam", learning_rate=0.001, decay_rate=1, decay_step=30000)
trainer.train_unet(data_provider, "/data/Cell/yunzhe/new_data_more/", training_iters=10000, epochs=101,
                   save_epoch=1, restore=True, verify_epoch=1, display_step=1000)

# data_provider = patch_provider.PatchProvider("/data/Cell/norm_data/training_data/",
#                                              "/data/Cell/norm_data/test_data/", data_augmentation=True, resample=True,
#                                              train_percent=0.8, sample_number=10000, sample_size=51)
#
# simpleNet = SimpleNet(sample_size=s m51)
# trainer = Trainer(simpleNet, batch_size=256, optimizer="adam",
#                   learning_rate=0.001, decay_rate=1, decay_step=10000, momentum=0.5)
# trainer.train_classification(data_provider, "/home/cell/yunzhe/miccai_code/classify_model", training_iters=1000, epochs=101,
#                              save_epoch=5, restore=True, verify_epoch=1, display_step=100)