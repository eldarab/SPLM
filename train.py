import time
from typing import Union

import torch
from torch import nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from optim import SPLM
from utils.supported_experiments import ADAM
from utils.supported_experiments import SPLM as SPLM_NAME
from utils.utils import report_metrics


def trainer(
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Union[SPLM, Adam],
        scheduler,  # _BetaScheduler / _LRScheduler
        metrics_fns: dict,
        params
) -> dict:
    # initialization
    metrics_values = {'train_' + metric_name: [] for metric_name in metrics_fns.keys()}
    metrics_values.update({'eval_' + metric_name: [] for metric_name in metrics_fns.keys()})
    metrics_values['train_loss'] = []
    metrics_values['eval_loss'] = []
    metrics_values['epoch_time'] = []

    classes = getattr(train_loader.dataset, 'classes')

    if 'beta' in optimizer.param_groups[0]:
        metrics_values['beta'] = []

    for epoch in range(params['optim']['epochs']):
        t = time.time()
        for i, (x, y) in tqdm(enumerate(train_loader), desc=f'epoch {epoch:<3d}'):
            if torch.cuda.is_available() and params['general']['use_cuda']:
                x = x.cuda()
                y = y.cuda()

            # forward pass
            output = model(x)

            # compute loss
            loss = loss_fn(output, y) / len(train_loader)

            # optimization step
            optimizer.zero_grad()
            if params['optim']['optimizer'] == SPLM_NAME:
                # loss.backward() --- NO NEED TO DO THAT WITH SPLM!!!
                optimizer.step(model=model, output=output, classes=classes, y_true=y)
            elif params['optim']['optimizer'] == ADAM:
                loss.backward()
                optimizer.step()

        metrics_values['epoch_time'].append(time.time() - t)
        if 'beta' in optimizer.param_groups[0]:
            metrics_values['beta'].append(optimizer.param_groups[0]['beta'])

        # evaluate
        metrics_values = evaluator(model, train_loader, loss_fn, metrics_fns, metrics_values, params)
        model.train(False)
        metrics_values = evaluator(model, eval_loader, loss_fn, metrics_fns, metrics_values, params)
        model.train(True)

        report_metrics(metrics_values)

        # scheduler step
        if scheduler:
            scheduler.step()

    return metrics_values


def evaluator(
        model,
        dataloader,
        loss_fn,
        metrics_fns,
        metrics_values,
        params
):
    """
    Evaluate a model without gradient calculation
    :param params:
    :type params:
    :param metrics_fns:
    :type metrics_fns:
    :param metrics_values:
    :type metrics_values:
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
    for metric_name, metric_fn in metrics_fns.items():
        metrics_values[mode + metric_name].append(0)

    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available() and params['general']['use_cuda']:
            x = x.cuda()
            y = y.cuda()

        output = model(x.view(params['optim']['batch_size'], params['model']['input_dim']))

        loss = loss_fn(output, y)
        for metric_name, metric_fn in metrics_fns.items():
            metrics_values[mode + metric_name][-1] += metric_fn(output.cpu().argmax(-1), y.cpu()) / len(dataloader)

    metrics_values[mode + 'loss'].append(loss.item())

    return metrics_values
