import torch
from torch import Tensor


def multiclass_hinge_loss(outputs: Tensor, targets: Tensor, margin=1.):
    # TODO change margin
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
        res = res - float(res[target]) + margin
        res[target] -= margin
        loss_tensor[i] = res.max()
    return loss_tensor.mean()


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
