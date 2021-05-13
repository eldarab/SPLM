import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from datasets import SyntheticDataset
from nets.fc import FCNet
from train import trainer
from utils.loss_functions import MulticlassHingeLoss
from sklearn.metrics import accuracy_score


def main_mnist(seed=42):
    # seed
    torch.manual_seed(seed)

    # data
    mnist_train = datasets.MNIST("./datasets", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

    # model
    model = FCNet(dims=[28 * 28, 128, 32, 10], activation='Softplus')


def main_synthetic(seed=42):
    torch.manual_seed(seed)

    # model parameters
    input_dim = 784
    hidden_dim = 32
    output_dim = num_classes = 10
    loss_fn = MulticlassHingeLoss(num_classes)

    # optimization parameters
    epochs = 100
    batch_size = 100
    K = 100
    beta = 1000.0
    use_cuda = False

    # synthetic data
    synthetic_train = SyntheticDataset(num_samples=1000, input_dim=input_dim, num_classes=num_classes)
    synthetic_eval = SyntheticDataset(num_samples=100, input_dim=input_dim, num_classes=num_classes)
    train_loader = DataLoader(synthetic_train, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(synthetic_eval, batch_size=batch_size, shuffle=False)

    # mnist data
    mnist_train = datasets.MNIST("./datasets", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test = datasets.MNIST("./datasets", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_train.data = mnist_train.data[:1000]
    mnist_test.data = mnist_test.data[:100]
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # initialize model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        nn.Softplus(),
        torch.nn.Linear(hidden_dim, output_dim),
    )

    metrics = trainer(
        model,
        train_loader,
        eval_loader,
        loss_fn,
        epochs=epochs,
        metrics_fns={'accuracy': accuracy_score},
        beta=beta,
        K=K,
        use_cuda=use_cuda,
    )

    for metric_name, metric_values in metrics.items():
        plt.plot(metric_values)
        plt.title(metric_name)
        plt.xlabel('Epochs')
        plt.show()


if __name__ == '__main__':
    main_synthetic(40)
