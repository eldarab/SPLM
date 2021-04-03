import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from nets.fc import FCNet


def main(seed=42):
    # seed
    torch.manual_seed(seed)

    # data
    mnist_train = datasets.MNIST("./datasets", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

    # model
    model = FCNet(dims=[28 * 28, 128, 32, 10], activation='Softplus')



if __name__ == '__main__':
    main()