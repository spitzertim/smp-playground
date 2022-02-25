from logging import root
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from natsort import natsorted
from functools import reduce
from skimage import io

from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


train_dataset = VOCSegmentation(root="data/", image_set="train")

idx_list = []
for i in tqdm(range(0, len(train_dataset))):
    # for i in tqdm(range(0, 4)):
    image, mask = train_dataset[i]
    mask = np.array(mask)
    mask = np.where(mask == 15, 1.0, 0.0)
    uniques = np.unique(mask)
    if 1.0 in uniques:
        idx_list.append(str(i))


print((idx_list))


with open("voc_ext_train_idxs.txt", "w") as f:
    f.write("\n".join(idx_list))
