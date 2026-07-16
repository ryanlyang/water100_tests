import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from colour import Color

from os.path import join as oj
import os
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
_data_root = os.path.join(_repo_root, "data")

mnist_trainset = datasets.MNIST(root=_data_root, train=True, download=True, transform=None)
color_x = np.zeros((60000, 1, 28, 28))
color_x = mnist_trainset.data[:, None].numpy().astype(np.float32)
color_y = mnist_trainset.targets.numpy().copy()

choice_1 = np.random.choice(2, size = len(color_x))*23
choice_2 = np.random.choice(2, size = len(color_x))*23
for i in range(len(color_x)):

    color_x[i, :, choice_1[i]:choice_1[i]+5, choice_2[i]:choice_2[i]+5] = 255- 25*color_y[i]
color_x /=color_x.max()
color_x = color_x*2 -1
np.save(oj(_data_root, "ColorMNIST", "train_x_decoy.npy"), color_x)

from os.path import join as oj
mnist_trainset = datasets.MNIST(root=_data_root, train=False, download=True, transform=None)
color_x = np.zeros((len(mnist_trainset.data), 1, 28, 28))
color_x = mnist_trainset.data[:, None].numpy().astype(np.float32)
color_y = mnist_trainset.targets.numpy().copy()
choice_1 = np.random.choice(2, size = len(color_x))*23
choice_2 = np.random.choice(2, size = len(color_x))*23
for i in range(len(color_x)):
    color_x[i, :, choice_1[i]:choice_1[i]+5, choice_2[i]:choice_2[i]+5] = 0+ 25*color_y[i]
color_x /=color_x.max()
color_x = color_x*2 -1

np.save(oj(_data_root, "ColorMNIST", "test_x_decoy.npy"), color_x)
