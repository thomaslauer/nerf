import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import imageio
imageio.plugins.freeimage.download()

import cv2
import PIL

from torch.utils.data import Dataset
from utils import makeHist

class MyData(Dataset): 

    def __init__(self, path):
        self.path = path
        self.transforms = T.ToTensor()

    def __getitem__(self, i):

        albedo = imageio.imread(f"{self.path}/inputs/albedo{i}.exr")
        color = imageio.imread(f"{self.path}/inputs/color{i}.exr")
        reference = imageio.imread(f"{self.path}/inputs/reference{i}.exr")

        output = np.concatenate([color, albedo], axis=2)
        print(output.shape)


        return self.transforms(output), self.transforms(reference)

    def __len__(self):
        # todo code this better
        return 60


# image = cv2.imread("data/classroom/inputs/albedo0.exr", cv2.IMREAD_ANYCOLOR)

# image = imageio.imread('data/sponza/inputs/color59.exr')

dataset = MyData("data/sponza")

x,y = dataset.__getitem__(0)

print(x.shape)
print(y.shape)