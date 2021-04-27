import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from datasets import SyntheticDataset
from nets.fc import FCNet
from train import multiclass_hinge_loss, trainer
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

    # model architecture
    input_dim = 784
    hidden_dim = 32
    output_dim = num_classes = 10
    p = input_dim * hidden_dim + hidden_dim * output_dim + hidden_dim + output_dim  # problem dim (num parameters)
    Y = list(range(num_classes))  # all labels

    # optimization parameters
    K = 100
    beta = 1.0

    # initialize data
    synthetic_train = SyntheticDataset(num_samples=1000, input_dim=input_dim, num_classes=num_classes)
    synthetic_eval = SyntheticDataset(num_samples=100, input_dim=input_dim, num_classes=num_classes)
    train_loader = DataLoader(synthetic_train, batch_size=100, shuffle=True)
    eval_loader = DataLoader(synthetic_eval, batch_size=100, shuffle=False)

    # initialize model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        nn.Softplus(),
        torch.nn.Linear(hidden_dim, output_dim),
        nn.Softplus()
    )

    trainer(
        model,
        train_loader,
        eval_loader,
        multiclass_hinge_loss,
        epochs=5,
        metrics_fns={'accuracy': accuracy_score},
        beta=beta,
        K=K,
        use_cuda=False
    )


if __name__ == '__main__':
    main_synthetic()
