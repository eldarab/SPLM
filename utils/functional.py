import torch
import torch.nn as nn


def multiclass_hinge_loss(outputs, targets, num_classes, margin):
    """

    :param outputs: classifier outputs (scores)
    :param target: index of target class
    :param num_classes:
    :return:
    """
    outputs += 1.0
    loss = - outputs[target]
