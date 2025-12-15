"""
Implementation adapted from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec


"""

import typing as tp

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRootLRScheduler(_LRScheduler):
    """Inverse square root LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        warmup_init_lr (tp.Optional[float]): Initial learning rate
            during warmup phase. When not set, use the provided learning rate.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, warmup_init_lr: tp.Optional[float] = 0):
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            warmup_init_lr = self.warmup_init_lr or 0
            lr_step = (lr - warmup_init_lr) / self.warmup_steps
            lr = warmup_init_lr + step * lr_step
        else:
            decay_factor = lr * self.warmup_steps**0.5
            lr = decay_factor * step**-0.5
        return lr

    def get_lr(self):
        return [self._get_sched_lr(base_lr, self._step_count) for base_lr in self.base_lrs]