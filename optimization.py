import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from utils.functional import simplex_projection, calc_lip_const, flatten_params, reshape_params


def inner_minimization(A: Tensor, b: Tensor, w_t: Tensor, beta: float, K: int = 50) -> Tensor:
    """
    Performs inner minimization of SPLM algorithm.
    Note: This is not a general implementation of FDPG! It is written specifically for the purposes of SPLM!
    Based on: https://web.iem.technion.ac.il/images/user-files/becka/papers/40.pdf
    :param A: [n x p] (in the case of SPLM these are the gradients stacked)
    :param b: [n x 1] (in the case of SPLM these are linear approximations stacked)
    :param w_t: [p x 1] ("current" net weights)
    :param beta: scalar (matrix M constant)
    :param K: (number of iterations to perform)
    :return: updated net weights w^{t+1}
    """
    # 0. Input
    n = A.size(0)
    p = A.size(1)
    w_t = w_t.view(p, 1)  # TODO necessary?
    A = A.view(n, p)
    b = b.view(n, 1)

    L = calc_lip_const(A, beta)

    # 1. Initialization
    y_k = - torch.ones(n, 1).fill_(1 / n)
    w_k_next = y_k.clone()
    t_k_next = 1

    # 2. Loop
    for k in range(K):
        w_k = w_k_next
        t_k = t_k_next
        y_k_prev = y_k

        u_k = w_t + (1 / beta) * A.t().mm(w_k)
        y_k = - simplex_projection((1 / L) * (A.mm(u_k) + b - L * w_k))
        t_k_next = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2
        w_k_next = y_k + ((t_k - 1) / t_k_next) * (y_k - y_k_prev)

    return u_k


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


def g_i_y_hat2(output: Tensor, y: int, y_hat: int):
    # Multiclass hinge loss w.r.t. only one label
    output = output.squeeze()
    zero_one_loss = 1.0 if y != y_hat else 0.0
    return zero_one_loss + output[y_hat] - output[y]


def prepare_fdpg_multiclass_classification(model: nn.Module, output: Tensor, classes: list, y_true: int) -> (Tensor, Tensor, Tensor):
    """
    Initializes the matrix A^t_y and the vector b^t_y in the case of hinge loss multiclass
    classification.

    :param model: The model to optimize.
    :type model:
    :param output:
    :type output:
    :param classes:
    :type classes:
    :param y_true:
    :type y_true:
    :return:
    :rtype:
    """
    # initialization
    model.zero_grad()  # TODO
    w = flatten_params(model.parameters())
    A, b = [], []

    for class_idx in range(len(classes)):
        # compute gradients w.r.t. each output coordinate, section 4.1 in the paper
        b_y_hat = g_i_y_hat(output, y_true, class_idx)
        b_y_hat.backward(create_graph=True)  # TODO create graph?
        a_y_hat = flatten_params(model.parameters(), flatten_grad=True)  # model gradients at iteration t

        # python technicalities
        A.append(a_y_hat.unsqueeze(0))
        b.append((b_y_hat - a_y_hat.dot(w)).unsqueeze(0))
        model.zero_grad()

    return w, torch.cat(A, dim=0), torch.cat(b, dim=0).unsqueeze(1)


def splm_step(model, output, classes, y, beta, K):
    # TODO choose matrix M
    w_t, A_i, b_i = prepare_fdpg_multiclass_classification(model, output, classes, y)
    with torch.no_grad():
        w_t_plus_1 = inner_minimization(A_i, b_i, w_t, beta, K)  # FDPG
        reshaped_params = reshape_params(model.parameters(), w_t_plus_1)
        for idx, param in enumerate(model.parameters()):
            param.copy_(reshaped_params[idx])
