import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm

class CNN_1(nn.Module):

    def __init__(self):
        super(CNN_1, self).__init__()
        # define the layer of network

        self.conv1 = nn.Conv2d(3, out_channels = 32, kernel_size = 3)
        self.maxpool1 = nn.MaxPool2d(kenel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv3 = nn.Conv2d(in_clannels = 128, out_channels = 128, kernel_size = 3)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(19*19*128, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(F.relu(x, inplace = True))
        x = self.conv2(x)
        x = F.relu(x, inplace = True)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x, inplace = True)
        x = x.view(-1, 19 * 19 * 128)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.sigmoid(x)
        return output

