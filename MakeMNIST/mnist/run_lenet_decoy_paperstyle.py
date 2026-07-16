from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
import numpy as np
import os
import sys
import pickle as pkl
from copy import deepcopy
import time
import random

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

sys.path.append(os.path.join(_here, "DecoyMNIST"))
from params_save import S

model_path = os.path.join(_repo_root, "models", "DecoyMNIST")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True


def save(p, out_name):
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


parser = argparse.ArgumentParser(description='PyTorch DecoyMNIST PNG Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--png-root', type=str, default=None,
                    help='root folder for PNGs (default: data/DecoyMNIST_png)')

args = parser.parse_args()
s = S(args.epochs)
s.regularizer_rate = 0
s.num_blobs = 0
s.seed = args.seed

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

png_root = args.png_root or oj(_repo_root, "data", "DecoyMNIST_png")
transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])

complete_dataset = ImageFolder(oj(png_root, "train"), transform=transform)

num_train = int(len(complete_dataset) * .9)
num_test = len(complete_dataset) - num_train
torch.manual_seed(0)
train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [num_train, num_test])

train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = utils.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_dataset = ImageFolder(oj(png_root, "test"), transform=transform)
val_loader = utils.DataLoader(val_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)


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


def test(args, model, device, test_loader, epoch, label="Val"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nEpoch {} {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, label, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    s.losses_test.append(test_loss)
    s.accs_test.append(100. * correct / len(test_loader.dataset))
    return test_loss


best_model_weights = None
best_test_loss = 100000
patience = 2
cur_patience = 0

start = time.time()

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test_loss = test(args, model, device, test_loader, epoch)
    if test_loss < best_test_loss:
        cur_patience = 0
        best_test_loss = test_loss
        best_model_weights = deepcopy(model.state_dict())
    else:
        cur_patience += 1
        if cur_patience > patience:
            break

end = time.time()
s.time_per_epoch = (end - start) / (epoch)

s.model_weights = best_model_weights
test(args, model, device, val_loader, epoch + 1, label="Test")
s.dataset = "Decoy"
s.method = "Vanilla"

np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s, pid)
