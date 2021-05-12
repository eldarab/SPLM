import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from datasets import SyntheticDataset
from plotting import plot_metrics
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

        train_loader = DataLoader(synthetic_train, batch_size=params['optim']['batch_size'], shuffle=True)
        eval_loader = DataLoader(synthetic_eval, batch_size=params['optim']['batch_size'], shuffle=False)

    elif params['data']['dataset'] == 'mnist':
        mnist_train = datasets.MNIST("./datasets", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_test = datasets.MNIST("./datasets", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        mnist_train.data = mnist_train.data[:params['data']['train_samples']]
        mnist_test.data = mnist_test.data[:params['data']['eval_samples']]

        train_loader = DataLoader(mnist_train, batch_size=params['optim']['batch_size'], shuffle=True)
        eval_loader = DataLoader(mnist_test, batch_size=params['optim']['batch_size'], shuffle=False)

    else:
        raise RuntimeError(f'Illegal dataset {params["data"]["dataset"]}')

    return train_loader, eval_loader


def init_loss(params: dict):
    if params['model']['loss'] == 'hinge':
        loss_fn = MulticlassHingeLoss(params['model']['num_classes'])
    elif params['model']['loss'] == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise RuntimeError(f'Illegal loss {params["model"]["loss"]}')
    return loss_fn


def main():
    with open('experiments/splm-mnist.yml') as f:
        params = yaml.safe_load(f)

    torch.manual_seed(params['general']['seed'])

    loss_fn = init_loss(params)

    train_loader, eval_loader = init_data(params)

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
