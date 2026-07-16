import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from tqdm import tqdm
from colour import Color
import os
from os.path import join as oj
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
_data_root = os.path.join(_repo_root, "data")
np.random.seed(0)
red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [np.asarray(x.get_rgb()) for x in colors]


mnist_trainset = datasets.MNIST(root=_data_root, train=True, download=True, transform=None)
num_train = int(len(mnist_trainset)*.9)
num_val = len(mnist_trainset)  - num_train 
torch.manual_seed(0);
train_dataset, val_dataset,= torch.utils.data.random_split(mnist_trainset, [num_train, num_val])

num_samples = len(train_dataset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = np.empty(num_samples, dtype = np.int16)
for i in tqdm(range(num_samples)):
    my_color  = colors[train_dataset.dataset.train_labels[train_dataset.indices[i]].item()]
    color_x[i ] = train_dataset.dataset.data[train_dataset.indices[i]].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
    color_y[i] = train_dataset.dataset.targets[train_dataset.indices[i]]
os.makedirs(oj(_data_root, "ColorMNIST"), exist_ok = True)
np.save(oj(_data_root, "ColorMNIST", "train_x.npy"), color_x)
np.save(oj(_data_root, "ColorMNIST", "train_y.npy"), color_y)


num_samples = len(val_dataset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = np.empty(num_samples, dtype = np.int16)
for i in tqdm(range(num_samples)):
    my_color  = colors[9-val_dataset.dataset.train_labels[val_dataset.indices[i]].item()]
    color_x[i ] = val_dataset.dataset.data[val_dataset.indices[i]].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
    color_y[i] = val_dataset.dataset.targets[val_dataset.indices[i]]
os.makedirs(oj(_data_root, "ColorMNIST"), exist_ok = True)
np.save(oj(_data_root, "ColorMNIST", "val_x.npy"), color_x)
np.save(oj(_data_root, "ColorMNIST", "val_y.npy"), color_y)




mnist_trainset = datasets.MNIST(root=_data_root, train=False, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)

color_y = mnist_trainset.train_labels.numpy().copy()
for i in tqdm(range(num_samples)):
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*colors[9-color_y[i]] [:, None, None]

np.save(oj(_data_root, "ColorMNIST", "test_x.npy"),  color_x)
np.save(oj(_data_root, "ColorMNIST", "test_y.npy"), color_y)
