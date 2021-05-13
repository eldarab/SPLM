import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
        self.num_samples = num_samples
        self.x = torch.randn(size=(num_samples, input_dim))
        self.y = torch.randint(low=0, high=num_classes - 1, size=(num_samples,))
        self.targets = [class_idx for class_idx in range(num_classes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
