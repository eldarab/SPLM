import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.fc import FCNet


class FeedForwardMNISTClassifier(nn.Module):
    def __init__(self):
        super(FeedForwardMNISTClassifier, self).__init__()
        self.model = FCNet(dims=[28 * 28, 32, 10], activation='Softplus')


class CnnMNISTClassifier(nn.Module):
    def __init__(self):
        super(CnnMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x