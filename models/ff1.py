import torch
import torch.nn as nn

from nets.fc import FCNet


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.model = FCNet(dims=[28*28, 128, 32, 10], activation='Softplus')

