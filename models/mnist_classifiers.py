import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.fc import FCNet


class FeedForwardMNISTClassifier(nn.Module):
    def __init__(self, activation='Softplus', num_classes=10):
        super(FeedForwardMNISTClassifier, self).__init__()
        self.num_classes = 10
        self.model = FCNet(dims=[28 * 28, 32, num_classes], activation=activation)
