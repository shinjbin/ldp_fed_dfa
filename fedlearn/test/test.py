# loading training data
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torchvision
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, Subset


train_ds = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

split_ds = []
length = len(train_ds)
subset = length // 5
for i in range(5):
    indices = list(range(subset*i, subset*(i+1)))
    split_ds.append(Subset(train_ds, indices))

for s in split_ds:
    print(len(s))

# shuffled_indices = torch.randperm(len(dataset))
# inputs = dataset.data[shuffled_indices]
# labels = dataset.targets[shuffled_indices]
#
# # partition data into num_clients
# split_size = len(dataset) // 5
# split_datasets = list(
#             zip(
#                 torch.split(inputs, split_size),
#                 torch.split(labels, split_size)
#             ))
#
# print(split_datasets[0])