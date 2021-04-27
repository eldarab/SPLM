import torch
from torch import nn as nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SyntheticDataset
from optimization import splm_step


def trainer(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader,
            loss_fn: nn.Module, epochs: int, metrics_fns: dict, beta: float, K: int, use_cuda=False):
    # TODO return type hinting

    # initialization
    metrics_values = {'train_' + metric_name: [] for metric_name in metrics_fns.keys()}
    metrics_values.update({'eval_' + metric_name: [] for metric_name in metrics_fns.keys()})
    metrics_values['train_loss'] = []
    metrics_values['eval_loss'] = []

    classes = train_loader.dataset.targets  # TODO
    # TODO model.init_weights_normal()

    for epoch in range(epochs):
        metrics_values['train_loss'].append(0)
        for i, (x, y) in enumerate(train_loader):
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            # forward pass
            output = model(x)

            # compute loss
            loss = loss_fn(output, y)
            metrics_values['train_loss'][-1] += loss / len(train_loader)

            # optimization step
            splm_step(model, output, classes, y, beta, K)

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

        output = model(x)

        loss = loss_fn(output, y)
        for metric_name, metric_fn in metrics_fns.items():
            metrics_values[mode + metric_name][-1] += metric_fn(output.argmax(-1), y) / dataloader.batch_size

    metrics_values[mode + 'loss'].append(loss)

    return metrics_values


def multiclass_hinge_loss(outputs: Tensor, targets: Tensor):
    # Implements the loss presented in this paper
    # https://www.jmlr.org/papers/volume2/crammer01a/crammer01a.pdf
    if not (targets.dtype == torch.int or
            targets.dtype == torch.int8 or
            targets.dtype == torch.int16 or
            targets.dtype == torch.int32 or
            targets.dtype == torch.int64):
        raise RuntimeError(f"Targets tensor dtype must be some integer type, got {targets.dtype}")
    if len(targets.size()) >= 2:
        raise RuntimeError(f"targets must have at most 1 dimensions. targets dim is instead {len(targets.size())}")
    if len(targets.size()) == 0:
        targets = targets.view(1, )
    if len(outputs.size()) == 1:
        outputs = outputs.view(1, outputs.nelement())
    if not (outputs.shape[0] == targets.shape[0]):
        raise RuntimeError(f"outputs and targets must have the same dim=0")

    # targets = targets.int()
    loss_tensor = torch.zeros(outputs.size(0))
    for i, (output, target) in enumerate(zip(outputs, targets)):
        res = output.clone()
        res = res - float(res[target]) + 1
        res[target] -= 1
        loss_tensor[i] = res.max()
    return loss_tensor.mean()
