import argparse
import os
import time
from pathlib import Path, PosixPath

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from datasets.synthetic import SyntheticDataset
from optim.splm import prepare_inner_minimization_multiclass_classification
from optim.beta_scheduler import StepBeta
from optim.splm import SPLM as SPLMOptimizer
from utils.logging import plot_metrics, plot_overflow, get_time_str
from train import trainer
from utils.paths import DATASETS_DIR, EXPERIMENTS_RESULTS_DIR
from utils.supported_experiments import *
from utils.functional import MulticlassHingeLoss
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
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(DATASETS_DIR, train=True, download=True, transform=transform)
        eval_dataset = datasets.MNIST(DATASETS_DIR, train=False, download=True, transform=transform)
        train_dataset.data = train_dataset.data[:train_samples]
        eval_dataset.data = eval_dataset.data[:eval_samples]
    elif dataset == CIFAR10:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(DATASETS_DIR, train=True, download=True, transform=transform)
        eval_dataset = datasets.CIFAR10(DATASETS_DIR, train=False, download=True, transform=transform)
        train_dataset.data = train_dataset.data[:train_samples]
        eval_dataset.data = eval_dataset.data[:eval_samples]
    else:
        raise RuntimeError(f'Dataset "{dataset}" is not supported. Supported datasets are: {SUPPORTED_DATASETS}.')

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


def init_optim(params: dict, model):
    optimizer = params['optim']['optimizer']
    scheduler_name = params['optim']['scheduler']['scheduler_name'] if params['optim']['scheduler']['use_scheduler'] else None

    if optimizer == ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['optim']['lr'])
    elif optimizer == SPLM:
        optimizer = SPLMOptimizer(
            params=model.parameters(),
            prepare_inner_minimization_fn=prepare_inner_minimization_multiclass_classification,
            K=int(params['optim']['K']),
            beta=float(params['optim']['beta'])
        )
    else:
        raise RuntimeError(f'Optimizer "{optimizer}" is not supported. Supported optimizers are: {SUPPORTED_OPTIMIZERS}.')

    if scheduler_name is None:
        scheduler = None
    elif scheduler_name == STEP_BETA:
        scheduler = StepBeta(
            optimizer=optimizer,
            step_size=params['optim']['scheduler']['step_size'],
            gamma=params['optim']['scheduler']['gamma'],
        )
    else:
        raise RuntimeError(f'Scheduler "{scheduler_name}" is not supported. Supported schedulers are {SUPPORTED_SCHEDULERS}.')

    return optimizer, scheduler


def init_results_dir(params: dict, config_path):
    """
    Initializes a results folder. Creates an
    :param params:
    :param config_path: PosixPath
    :return:
    """
    experiment_name = f"{params['model']['model_name']}_{params['optim']['optimizer']}_{params['data']['dataset']}__{get_time_str()}"
    results_dir = EXPERIMENTS_RESULTS_DIR / config_path.parent.name / experiment_name
    os.makedirs(str(results_dir))
    with open(f'{results_dir}/config.yml', 'w') as f:
        yaml.safe_dump(params, f)
    print(f'Successfully initialized results dir {results_dir}')
    return results_dir


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
    parser.add_argument("--file", type=str, required=True, help="path to config .yml file.")
    args = parser.parse_args()
    args.file = Path(args.file)
    with open(str(args.file)) as f:
        params = yaml.safe_load(f)

    # set seed
    torch.manual_seed(params['general']['seed'])

    # init experiment
    results_dir = init_results_dir(params, args.file)
    loss_fn = init_loss(params)
    train_loader, eval_loader = init_data(params)
    model = init_model(params)
    if torch.cuda.is_available() and params['general']['use_cuda']:
        model.to('cuda')
    optimizer, scheduler = init_optim(params, model)

    # run experiment
    try:
        metrics = trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_fns={'accuracy': accuracy_score},
            params=params,
        )
        plot_metrics(metrics, results_dir)
    except OverflowError:
        print('Reporting overflow and exiting.')
        plot_overflow(results_dir)
        exit(0)


if __name__ == '__main__':
    main()
