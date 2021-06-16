import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from datasets import SyntheticDataset
from nets.fc import FCNet
from plotting import plot_metrics, report_overflow
from train import trainer
from utils.loss_functions import MulticlassHingeLoss


def init_data(params: dict):
    if params['data']['dataset'] == 'synthetic':
        synthetic_train = SyntheticDataset(num_samples=params['data']['train_samples'],
                                           input_dim=params['model']['input_dim'],
                                           num_classes=params['model']['num_classes'])
        synthetic_eval = SyntheticDataset(num_samples=params['data']['eval_samples'],
                                          input_dim=params['model']['input_dim'],
                                          num_classes=params['model']['num_classes'])

        train_loader = DataLoader(synthetic_train, batch_size=params['optim']['batch_size'], shuffle=True, drop_last=True)
        eval_loader = DataLoader(synthetic_eval, batch_size=params['optim']['batch_size'], shuffle=False, drop_last=True)

    elif params['data']['dataset'] == 'mnist':
        mnist_train = datasets.MNIST("./datasets", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_test = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_train.data = mnist_train.data[:params['data']['train_samples']]
        mnist_test.data = mnist_test.data[:params['data']['eval_samples']]

        train_loader = DataLoader(mnist_train, batch_size=params['optim']['batch_size'], shuffle=True, drop_last=True)
        eval_loader = DataLoader(mnist_test, batch_size=params['optim']['batch_size'], shuffle=False, drop_last=True)

    else:
        raise RuntimeError(f'Illegal dataset {params["data"]["dataset"]}')

    return train_loader, eval_loader


def init_loss(params: dict):
    if params['model']['loss'] == 'hinge':
        margin = params['model']['margin'] if 'margin' in params['model'] else 1.
        loss_fn = MulticlassHingeLoss(margin)
    elif params['model']['loss'] == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif params['model']['loss'] == 'MultiMarginLoss':
        margin = params['model']['margin'] if 'margin' in params['model'] else 1.
        loss_fn = nn.MultiMarginLoss(margin=margin)
    else:
        raise RuntimeError(f'Illegal loss {params["model"]["loss"]}')
    return loss_fn


def init_experiment_folder(params: dict):
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    title = f"{params['optim']['optimizer']}_optimizer__{params['model']['loss']}_loss"
    os.makedirs(f'./figs/{title}__{time_str}')
    with open(f'./figs/{title}__{time_str}/params.yml', 'w') as f2:
        yaml.safe_dump(params, f2)
    return time_str, title


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="path to .yml file for experiments.")
    args = parser.parse_args()

    with open(args.file) as f:
        params = yaml.safe_load(f)

    torch.manual_seed(params['general']['seed'])

    time_str, title = init_experiment_folder(params)

    loss_fn = init_loss(params)
    train_loader, eval_loader = init_data(params)  # TODO normalize data

    model = FCNet(dims=[params['model']['input_dim'], params['model']['hidden_dim'], params['model']['output_dim']],
                  activation=params['model']['activation'])

    if torch.cuda.is_available() and params['general']['use_cuda']:
        model.to('cuda')

    try:
        metrics = trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_fn=loss_fn,
            metrics_fns={'accuracy': accuracy_score},
            params=params,
        )
        plot_metrics(metrics, time_str, title)
    except OverflowError:
        print('Reporting overflow and exiting.')
        report_overflow(time_str, title)
        exit(0)


if __name__ == '__main__':
    main()
