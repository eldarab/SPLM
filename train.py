import time

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimization import prepare_inner_minimization_multiclass_classification, SPLM
from utils.utils import report_metrics


def trainer(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, loss_fn: nn.Module, metrics_fns: dict, params):
    # TODO return type hinting

    # initialization
    metrics_values = {'train_' + metric_name: [] for metric_name in metrics_fns.keys()}
    metrics_values.update({'eval_' + metric_name: [] for metric_name in metrics_fns.keys()})
    metrics_values['train_loss'] = []
    metrics_values['eval_loss'] = []
    metrics_values['epoch_time'] = []

    classes = [i for i in range(params['model']['num_classes'])]  # TODO
    # TODO model.init_weights_normal()

    if params['optim']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['optim']['lr'])
    else:
        optimizer = SPLM(
            params=model.parameters(),
            prepare_inner_minimization_fn=prepare_inner_minimization_multiclass_classification,
            K=params['optim']['K'],
            beta=params['optim']['beta']
        )

    for epoch in range(params['optim']['epochs']):
        t = time.time()
        for i, (x, y) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available() and params['general']['use_cuda']:
                x = x.cuda()
                y = y.cuda()

            # forward pass
            output = model(x.view(params['optim']['batch_size'], params['model']['input_dim']))

            # compute loss
            # TODO ?

            # optimization step
            if params['optim']['optimizer'] == 'splm':
                optimizer.step(model=model, output=output, classes=classes, y_true=y)
            elif params['optim']['optimizer'] == 'adam':
                loss = loss_fn(output, y) / len(train_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics_values['epoch_time'].append(time.time() - t)

        # evaluate
        metrics_values = evaluator(model, train_loader, loss_fn, metrics_fns, metrics_values, params)
        model.train(False)
        metrics_values = evaluator(model, eval_loader, loss_fn, metrics_fns, metrics_values, params)
        model.train(True)

        report_metrics(epoch, metrics_values)

    return metrics_values


def evaluator(model, dataloader, loss_fn, metrics_fns, metrics_values, params):
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
        if torch.cuda.is_available() and params['general']['use_cuda']:
            x = x.cuda()
            y = y.cuda()

        output = model(x.view(params['optim']['batch_size'], params['model']['input_dim']))

        loss = loss_fn(output, y)
        for metric_name, metric_fn in metrics_fns.items():
            metrics_values[mode + metric_name][-1] += metric_fn(output.cpu().argmax(-1), y.cpu()) / len(dataloader)

    metrics_values[mode + 'loss'].append(loss.item())

    return metrics_values
