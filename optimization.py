import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from utils.functional import simplex_projection, calc_lip_const, flatten_params, reshape_params, g_i_y_hat


class SPLM(Optimizer):
    """
    Implements SPLM algorithm.

    """

    def __init__(self, params, prepare_inner_minimization_fn, beta=500., K=50):
        if not K > 0:
            raise ValueError(f'Invalid K (inner minimization steps): {K}')
        if not beta > 0.0:
            raise ValueError(f'Invalid beta: {beta}')

        defaults = dict(beta=beta, K=K)

        self.prepare_inner_minimization_fn = prepare_inner_minimization_fn
        self.beta = beta
        self.K = K

        super(SPLM, self).__init__(params, defaults)

        # TODO Support more than 1 param group
        if len(self.param_groups) > 1:
            raise NotImplementedError(f'This optimizer still doesnt support more than 1 param group.')

    def step(self, **inner_minimization_kwargs):
        """
        Performs a single optimization step.

        :return:
        :rtype:
        """

        # TODO choose matrix M
        w_t, A_i, b_i = self.prepare_inner_minimization_fn(**inner_minimization_kwargs)

        with torch.no_grad():
            w_t_plus_1 = self.__fdpg(A_i, b_i, w_t)
            reshaped_params = reshape_params(self.param_groups[0], w_t_plus_1)
            for idx, param in enumerate(self.param_groups[0]['params']):
                param.copy_(reshaped_params[idx])

    def __fdpg(self, A: Tensor, b: Tensor, w_t: Tensor) -> Tensor:
        """
        Performs inner minimization of SPLM algorithm.

        Note: This is not a general implementation of FDPG! It is written specifically for the purposes of SPLM!
        Based on: https://web.iem.technion.ac.il/images/user-files/becka/papers/40.pdf

        :param A: [n x p] (in the case of SPLM these are the gradients stacked)
        :param b: [n x 1] (in the case of SPLM these are linear approximations stacked)
        :param w_t: [p x 1] ("current" net weights)
        :return: updated net weights w^{t+1}
        """

        # 0. Input
        n = A.shape[0]
        p = A.shape[1]
        w_t = w_t.view(p, 1)  # TODO necessary?
        A = A.view(n, p)
        b = b.view(n, 1)

        L = calc_lip_const(A, self.beta)

        # 1. Initialization
        u_k = None
        y_k = - torch.ones(n, 1).fill_(1 / n)
        t_k_next = 1
        w_k_next = y_k.clone()

        # 2. Loop
        for k in range(self.K):
            w_k = w_k_next
            t_k = t_k_next
            y_k_prev = y_k

            u_k = w_t + (1 / self.beta) * A.t().mm(w_k)
            y_k = - simplex_projection((1 / L) * (A.mm(u_k) + b - L * w_k))
            t_k_next = (1 + np.sqrt(1 + 4 * (t_k ** 2))) / 2
            w_k_next = y_k + ((t_k - 1) / t_k_next) * (y_k - y_k_prev)

        return u_k


def prepare_inner_minimization_multiclass_classification(
        model: nn.Module,
        output: Tensor,
        classes: list,
        y_true: Tensor) -> (Tensor, Tensor, Tensor):
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
