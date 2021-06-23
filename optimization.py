import warnings
import weakref
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from utils.functional import simplex_projection, calc_lip_const, flatten_params, reshape_params, g_i_y_hat

# from torch.optim.lr_scheduler import StepLR

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


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
        y_k = - torch.ones(n, 1, device='cuda').fill_(1 / n)
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


class _BetaScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base betas
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_beta', group['beta'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_beta' not in group:
                    raise KeyError("param 'initial_beta' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_betas = [group['initial_beta'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `beta_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self._last_beta = [group['beta'] for group in self.optimizer.param_groups]

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_beta(self):
        """ Return last computed beta by current scheduler.
        """
        return self._last_beta

    def get_beta(self):
        # Compute beta using chainable form of the scheduler
        raise NotImplementedError

    @staticmethod
    def print_beta(is_verbose, group, beta, epoch=None):
        """Display the current beta.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting beta'
                      ' of group {} to {:.4e}.'.format(group, beta))
            else:
                print('Epoch {:5d}: adjusting beta'
                      ' of group {} to {:.4e}.'.format(epoch, group, beta))

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after beta scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`beta_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first beta_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `beta_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `beta_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the beta schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_beta_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_beta_called_within_step = True
                return self

            def __exit__(self, call_type, value, traceback):
                self.o._get_beta_called_within_step = False

        with _enable_get_beta_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_beta()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_beta"):
                    values = self._get_closed_form_beta()
                else:
                    values = self.get_beta()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, beta = data
            param_group['beta'] = beta
            self.print_beta(self.verbose, i, beta, epoch)

        self._last_beta = [group['beta'] for group in self.optimizer.param_groups]


class StepBeta(_BetaScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepBeta, self).__init__(optimizer, last_epoch, verbose)

    def get_beta(self):
        if not self._get_beta_called_within_step:
            warnings.warn("To get the last beta computed by the scheduler, "
                          "please use `get_last_beta()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['beta'] for group in self.optimizer.param_groups]
        return [group['beta'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_beta(self):
        return [base_beta * self.gamma ** (self.last_epoch // self.step_size)
                for base_beta in self.base_betas]


def prepare_inner_minimization_multiclass_classification(
        model: nn.Module,
        output: Tensor,
        classes: list,
        y_true: Tensor
) -> (Tensor, Tensor, Tensor):
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


def test_scheduler():
    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SPLM(
        params=model,
        prepare_inner_minimization_fn=prepare_inner_minimization_multiclass_classification,
    )
    scheduler = StepBeta(optimizer, step_size=10, gamma=10)

    for epoch in range(100):
        # optimizer.step()
        scheduler.step()
        print(f'epoch: {epoch} beta: {optimizer.param_groups[0]["beta"]}')


if __name__ == '__main__':
    test_scheduler()
