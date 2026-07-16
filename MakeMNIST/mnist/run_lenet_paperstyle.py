#from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
import numpy as np
import os
import sys
import pickle as pkl
from copy import deepcopy
import random

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

sys.path.append(os.path.join(_here, "ColorMNIST"))
from model import Net
from params_save import S

model_path = os.path.join(_repo_root, "models", "ColorMNIST_test")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True

def save(p, out_name):
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))


parser = argparse.ArgumentParser(description='PyTorch ColorMNIST PNG Training')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--png-root', type=str, default=None,
                    help='root folder for PNGs (default: data/ColorMNIST_png)')

args = parser.parse_args()
s = S(args.epochs)
s.regularizer_rate = 0
s.num_blobs = 0
s.seed = args.seed

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn': np.random.seed(12)} if use_cuda else {}


def _compute_mean_std(dataset, batch_size=512):
    loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0
    sum_ = torch.zeros(3)
    sumsq = torch.zeros(3)
    for data, _ in loader:
        b = data.size(0)
        total += b * data.size(2) * data.size(3)
        sum_ += data.sum(dim=(0, 2, 3))
        sumsq += (data ** 2).sum(dim=(0, 2, 3))
    mean = sum_ / total
    std = torch.sqrt(sumsq / total - mean ** 2)
    return mean, std


png_root = args.png_root or oj(_repo_root, "data", "ColorMNIST_png")
base_transform = Compose([ToTensor(), Lambda(lambda x: x * 255.0)])

train_dataset_raw = ImageFolder(oj(png_root, "train"), transform=base_transform)
val_dataset_raw = ImageFolder(oj(png_root, "val"), transform=base_transform)
test_dataset_raw = ImageFolder(oj(png_root, "test"), transform=base_transform)

mean, std = _compute_mean_std(train_dataset_raw)
norm = Normalize(mean.tolist(), std.tolist())
full_transform = Compose([base_transform, norm])

train_dataset = ImageFolder(oj(png_root, "train"), transform=full_transform)
val_dataset = ImageFolder(oj(png_root, "val"), transform=full_transform)
test_dataset = ImageFolder(oj(png_root, "test"), transform=full_transform)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = utils.DataLoader(val_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
test_loader = utils.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), weight_decay=0.001)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)
            acc = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
            s.losses_train.append(loss.item())
            s.accs_train.append(acc)
            s.cd.append(0.0)


def test(args, model, device, dataset_loader, is_test=False, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataset_loader.dataset)

    if is_test:
        s.acc_test = 100. * correct / len(dataset_loader.dataset)
        s.loss_test = test_loss
        if epoch is not None:
            print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, test_loss, correct, len(dataset_loader.dataset),
                100. * correct / len(dataset_loader.dataset)))
        else:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(dataset_loader.dataset),
                100. * correct / len(dataset_loader.dataset)))
    else:
        s.losses_dev.append(test_loss)
        s.accs_dev.append(100. * correct / len(dataset_loader.dataset))
        if epoch is not None:
            print('Epoch {} Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, test_loss, correct, len(dataset_loader.dataset),
                100. * correct / len(dataset_loader.dataset)))
        else:
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(dataset_loader.dataset),
                100. * correct / len(dataset_loader.dataset)))
    return test_loss


best_model_weights = None
best_test_loss = 100000

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test_loss = test(args, model, device, val_loader, epoch=epoch)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_weights = deepcopy(model.state_dict())

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

s.dataset = "Color"
test(args, model, device, test_loader, is_test=True, epoch=args.epochs)
s.method = "Vanilla"

np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s, pid)
