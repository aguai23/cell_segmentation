import openslide
from staintools import MacenkoNormalizer
from staintools.utils.visual import read_image, show
import scipy.misc as mc
import os
dir = "/home/yunzhe/Downloads/MoNuSeg Training Data/test_data/"
source_file = "/home/yunzhe/Downloads/MoNuSeg Training Data/training_data/TCGA-A7-A13F-01Z-00-DX1.png"
save_dir = "/home/yunzhe/norm_data/test_data/"
source = read_image(source_file)
normalizer = MacenkoNormalizer()
normalizer.fit(source)
for filename in os.listdir(dir):
    if filename.endswith(".png"):
        target = read_image(dir + filename)
        if dir + filename == source_file:
            mc.imsave(save_dir + filename, target)
            continue
        norm_target = normalizer.transform(target)
        show(norm_target, fig_size=(4,4))
        mc.imsave(save_dir + filename, norm_target)
