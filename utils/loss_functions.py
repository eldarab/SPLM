import torch
import torch.nn as nn
from torch import Tensor
from utils.functional import multiclass_hinge_loss


class MulticlassHingeLoss(nn.Module):
    def __init__(self, num_classes, margin=1):
        super(MulticlassHingeLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return multiclass_hinge_loss(inputs, targets)
