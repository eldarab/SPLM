import torch
from torch import Tensor


def multiclass_hinge_loss(outputs: Tensor, targets: Tensor, margin=1., reduction='mean') -> Tensor:
    assert outputs.shape[0] == targets.shape[0]
    batch_size = outputs.shape[0]
    num_classes = outputs.shape[1]

    # TODO to be revisited when PyTorch implement https://www.tensorflow.org/api_docs/python/tf/map_fn
    loss = torch.tensor(0.)
    for x, y in zip(outputs, targets):
        loss += (torch.relu(margin + x - x[y]).sum() - margin)
    loss /= num_classes

    if reduction == 'mean':
        loss /= batch_size

    return loss


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
    for param in params:
        reshaped_params.append(params_flattened[total_elements:total_elements + param.nelement()])
        reshaped_params[-1] = reshaped_params[-1].view_as(param)
        total_elements += param.nelement()
    return reshaped_params


def g_i_y_hat(output: Tensor, y_true: Tensor, y_hat: int):  # batch compatible version
    """

    :param output:
    :type output:
    :param y_true: y_true "target"
    :type y_true:
    :param y_hat:
    :type y_hat:
    :return:
    :rtype:
    """
    zero_one_loss = (~torch.eq(output.argmax(-1), y_true)).float().squeeze(0)
    loss = torch.tensor(0.)
    for idx, y in enumerate(y_true):
        loss += zero_one_loss[idx] + output[idx][y_hat] - output[idx][y]
    return loss / len(output)