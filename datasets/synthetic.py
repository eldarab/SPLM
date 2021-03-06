import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
        # default input_dim inspired by MNIST 784=28x28
        self.num_samples = num_samples
        self.x = torch.randn(size=(num_samples, input_dim))
        self.y = torch.randint(low=0, high=num_classes - 1, size=(num_samples,))
        self.classes = list(range(num_classes))
        self.class_to_idx = {str(v): v for v in self.classes}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
