import torch
import torch.nn as nn
from torch import Tensor


def multiclass_hinge_loss(outputs, targets, num_classes, margin):
    """

    :param outputs: classifier outputs (scores)
    :param target: index of target class
    :param num_classes:
    :return:
    """
    outputs += 1.0
    loss = - outputs[target]


def simplex_projection(v: Tensor) -> Tensor:
    # Performs projection of vector v (in R^n) onto the n-dimensional simplex.
    x = v.clone().flatten()
    n = x.size(0)
    if n != v.nelement():
        raise ValueError(f"v must be have a shape of (n, 1) or (n, ) and not {v.shape}")
    sorted_x = torch.sort(x, dim=0, descending=True)[0]
    elem_sum = 0
    delta = 0

    for i in range(n):
        elem_sum += sorted_x[i]
        delta = (1 - elem_sum) / (i + 1)
        if i + 1 == n or -delta >= sorted_x[i + 1]:
            break

    x += delta
    x[torch.lt(x, 0)] = 0
    x = x.view_as(v)
    return x


def calc_lip_const(A: Tensor, beta: float):
    # inner minimization utility function
    # TODO very strange how this works
    try:
        _, s, _ = A.svd()
        L = ((max(s).item()) ** 2) / beta
    except RuntimeError:  # torch SVD may not find a solution
        L = 1e-5
    if L == 0:
        L = 1e-5
    return L


def flatten_params(params, flatten_grad=False):
    """
    some description

    :param params: model params
    :type params:
    :param flatten_grad: Whether or not to flatten the gradients of the parameters
    :type flatten_grad: bool
    :return:
    :rtype: Tensor
    """
    if flatten_grad:
        return torch.cat([param.grad.flatten() for param in params])
    else:
        return torch.cat([param.flatten() for param in params])


def reshape_params(params, params_flattened):
    reshaped_params = []
    total_elements = 0
    params = [p for p in params]
    for param in params:
        reshaped_params.append(params_flattened[total_elements:total_elements + param.nelement()])
        reshaped_params[-1] = reshaped_params[-1].view_as(param)
        total_elements += param.nelement()
    return reshaped_params
