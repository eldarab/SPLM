import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from datasets import SyntheticDataset
from nets.fc import FCNet
from plotting import plot_metrics
from train import trainer
from utils.loss_functions import MulticlassHingeLoss
from sklearn.metrics import accuracy_score
import yaml


def main():
    with open('experiments/splm-mnist.yml') as f:
        params = yaml.load(f)

    torch.manual_seed(params['general']['seed'])

    # model parameters
    loss_fn = MulticlassHingeLoss(params['model']['num_classes'])
    # loss_fn = nn.CrossEntropyLoss()

    # initialize data
    batch_size = params['optim']['batch_size']
    if params['dataset']['name'] == 'synthetic':
        synthetic_train = SyntheticDataset(num_samples=1000, input_dim=params['model']['input_dim'], num_classes=params['model']['num_classes'])
        synthetic_eval = SyntheticDataset(num_samples=100, input_dim=params['model']['input_dim'], num_classes=params['model']['num_classes'])
        train_loader = DataLoader(synthetic_train, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(synthetic_eval, batch_size=batch_size, shuffle=False)

    elif params['dataset']['name'] == 'mnist':
        mnist_train = datasets.MNIST("./datasets", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_test = datasets.MNIST("./datasets", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_train.data = mnist_train.data[:1000]
        mnist_test.data = mnist_test.data[:100]
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # initialize model
    model = torch.nn.Sequential(
        torch.nn.Linear(params['model']['input_dim'], params['model']['hidden_dim']),
        nn.Softplus(),
        torch.nn.Linear(params['model']['hidden_dim'], params['model']['output_dim']),
    )

    metrics = trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        loss_fn=loss_fn,
        metrics_fns={'accuracy': accuracy_score},
        params=params,
    )

    plot_metrics(metrics, title=params['optim']['optimizer'])


if __name__ == '__main__':
    main()
