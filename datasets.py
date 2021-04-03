from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    mnist_train = datasets.MNIST("./datasets", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
    print()
