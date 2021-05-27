import torch
import torch.nn as nn
from torch import Tensor
from utils.functional import multiclass_hinge_loss


class MulticlassHingeLoss(nn.Module):
    def __init__(self, margin=1., reduction='mean'):
        super(MulticlassHingeLoss, self).__init__()
        self.margin = margin
        if reduction == 'mean' or reduction == 'sum':
            self.reduction = reduction
        else:
            raise RuntimeError(f'Unsupported reduction: "{reduction}"')

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return multiclass_hinge_loss(inputs, targets, self.margin, self.reduction)
