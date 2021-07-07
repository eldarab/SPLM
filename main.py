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

from datasets.synthetic import SyntheticDataset
from plotting import plot_metrics, report_overflow
from train import trainer
from utils.paths import DATASETS_DIR
from utils.supported_experiments import *
from utils.loss_functions import MulticlassHingeLoss
from models.mnist_classifiers import FeedForwardMNISTClassifier
from torchvision.models import vgg11_bn, resnet18


def init_data(params: dict):
    dataset = params['data']['dataset']
    train_samples = params['data']['train_samples']
    eval_samples = params['data']['eval_samples']
    batch_size = params['optim']['batch_size']

    if dataset == SYNTHETIC:
        train_dataset = SyntheticDataset(num_samples=train_samples, num_classes=params['model']['num_classes'])
        eval_dataset = SyntheticDataset(num_samples=eval_samples, num_classes=params['model']['num_classes'])
    elif dataset == MNIST:
        train_dataset = datasets.MNIST(DATASETS_DIR, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        eval_dataset = datasets.MNIST(DATASETS_DIR, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_dataset.data = train_dataset.data[:train_samples]
        eval_dataset.data = eval_dataset.data[:eval_samples]
    elif dataset == CIFAR10:
        train_dataset = datasets.CIFAR10(DATASETS_DIR, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        eval_dataset = datasets.CIFAR10(DATASETS_DIR, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_dataset.data = train_dataset.data[:train_samples]
        eval_dataset.data = eval_dataset.data[:eval_samples]
        raise NotImplementedError('First we need to normalize data!')
    else:
        raise RuntimeError(f'Illegal dataset {params["data"]["dataset"]}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, eval_loader


def init_loss(params: dict):
    loss = params['model']['loss']
    margin = params['model'].pop('margin', 1.)

    if loss == HINGE_LOSS:
        return MulticlassHingeLoss(margin=margin)
    elif loss == CE_LOSS:
        return nn.CrossEntropyLoss()
    elif loss == MULTI_MARGIN_LOSS:
        return nn.MultiMarginLoss(margin=margin)
    else:
        raise RuntimeError(f'Loss function "{loss}" is not supported. Supported losses are: {SUPPORTED_LOSSES}.')


def init_experiment_folder(params: dict):
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    experiment_name = f"{params['model']['model_name']}_{params['optim']['optimizer']}_{params['data']['dataset']}"
    os.makedirs(f'./figs/{experiment_name}__{time_str}')
    with open(f'./figs/{experiment_name}__{time_str}/params.yml', 'w') as f:
        yaml.safe_dump(params, f)
    return time_str, experiment_name


def init_model(params: dict):
    model_name = params['model']['model_name']

    if model_name == FF_MNIST_CLASSIFIER:
        return FeedForwardMNISTClassifier(activation=params['model']['activation'], num_classes=params['model']['num_classes'])
    elif model_name == VGG11_BN:
        return vgg11_bn(pretrained=params['model']['pretrained'], num_classes=params['model']['num_classes'])
    elif model_name == RESNET18:
        return resnet18(pretrained=params['model']['pretrained'], num_classes=params['model']['num_classes'])
    else:
        raise RuntimeError(f'Model "{model_name}" is not supported. Supported models are: {SUPPORTED_MODELS}.')


def main():
    # load experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="path to .yml file for experiments.")
    args = parser.parse_args()

    with open(args.file) as f:
        params = yaml.safe_load(f)

    torch.manual_seed(params['general']['seed'])

    # init experiment
    time_str, experiment_name = init_experiment_folder(params)
    loss_fn = init_loss(params)
    train_loader, eval_loader = init_data(params)
    model = init_model(params)

    if torch.cuda.is_available() and params['general']['use_cuda']:
        model.to('cuda')

    # run experiment
    try:
        metrics = trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_fn=loss_fn,
            metrics_fns={'accuracy': accuracy_score},
            params=params,
        )
        plot_metrics(metrics, time_str, experiment_name)
    except OverflowError:
        print('Reporting overflow and exiting.')
        report_overflow(time_str, experiment_name)
        exit(0)


if __name__ == '__main__':
    main()
