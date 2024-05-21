"""
CNN model for FEMNIST Dataset.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, channel_1=32, channel_2=64, num_classes=62, feature_dim=-1, use_batchnorm=False):
        super(SimpleCNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(1, channel_1, (5, 5))
        self.conv2 = nn.Conv2d(channel_1, channel_2, (5, 5))
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(channel_1)
        if feature_dim > 0:
            self.pre_fc = nn.Sequential(nn.Flatten(), nn.Linear(16 * channel_2, feature_dim))
            self.fc = nn.Linear(feature_dim, num_classes)
        else:
            self.pre_fc = nn.Flatten()
            self.fc = nn.Linear(16 * channel_2, num_classes)

    def forward(self, x, return_features=False):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.use_batchnorm:
            out = self.bn(out)
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        features = self.pre_fc(out)
        return features if return_features else self.fc(features)


class Cifar10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.main(x)


def simplecnn(pretrained=False, num_classes=62):
    return SimpleCNN(num_classes=num_classes)


def mini_simplecnn(pretrained=False, num_classes=62):
    return SimpleCNN(num_classes=num_classes, channel_1=10, channel_2=20)

