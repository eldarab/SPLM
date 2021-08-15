import warnings
import weakref
from functools import wraps

from torch.optim import Optimizer

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class BetaScheduler(object):

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


class StepBeta(BetaScheduler):
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
