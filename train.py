import time

import torch
from torch import nn as nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SyntheticDataset
from optimization import splm_step


def trainer(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, loss_fn: nn.Module, metrics_fns: dict, params):
    # TODO return type hinting

    # initialization
    metrics_values = {'train_' + metric_name: [] for metric_name in metrics_fns.keys()}
    metrics_values.update({'eval_' + metric_name: [] for metric_name in metrics_fns.keys()})
    metrics_values['train_loss'] = []
    metrics_values['eval_loss'] = []
    metrics_values['epoch_time'] = []

    classes = [i for i in range(10)]  # TODO
    # TODO model.init_weights_normal()

    for epoch in range(params['optim']['epochs']):
        t = time.time()
        loss = 0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            if params['general']['use_cuda']:
                x = x.cuda()
                y = y.cuda()

            # forward pass
            output = model(x.view(100, 784))

            # compute loss
            # loss += loss_fn(output, y) / len(train_loader)

            # optimization step
            splm_step(model, output, classes, y, params['optim']['beta'], params['optim']['K'])

        metrics_values['epoch_time'].append(time.time() - t)
        # evaluate
        metrics_values = evaluator(model, train_loader, loss_fn, metrics_fns, metrics_values, use_cuda)
        model.train(False)
        metrics_values = evaluator(model, eval_loader, loss_fn, metrics_fns, metrics_values, use_cuda)
        model.train(True)

        print(f"epoch={epoch}")
        for k, v in metrics_values.items():
            print(f"\t\t{k}={v[-1]:.3f}")
        print()

    return metrics_values


def evaluator(model, dataloader, loss_fn, metrics_fns, metrics_values, use_cuda):
    """
    Evaluate a model without gradient calculation
    :param metrics_fns:
    :type metrics_fns:
    :param metrics_values:
    :type metrics_values:
    :param use_cuda:
    :type use_cuda:
    :param loss_fn:
    :type loss_fn:
    :param dataloader:
    :type dataloader:
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    loss = 0
    mode = 'train_' if model.training else 'eval_'
    m = len(dataloader.dataset)
    for metric_name, metric_fn in metrics_fns.items():
        metrics_values[mode + metric_name].append(0)

    for i, (x, y) in enumerate(dataloader):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        output = model(x.view(100, 784))

        loss = loss_fn(output, y)
        for metric_name, metric_fn in metrics_fns.items():
            metrics_values[mode + metric_name][-1] += metric_fn(output.argmax(-1), y) / len(dataloader)

    metrics_values[mode + 'loss'].append(loss.item())

    return metrics_values
