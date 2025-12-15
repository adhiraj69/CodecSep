
"""
Implementation adapted from SDCodec: https://github.com/XiaoyuBIE1994/SDCodec

Scheduler adapted and modified from AudioCraft project https://github.com/facebookresearch/audiocraft/tree/main
"""

# flake8: noqa
from .cosine_lr_scheduler import CosineLRScheduler
from .exponential_lr_scheduler import ExponentialLRScheduler
from .inverse_sqrt_lr_scheduler import InverseSquareRootLRScheduler
from .linear_warmup_lr_scheduler import LinearWarmupLRScheduler
from .polynomial_decay_lr_scheduler import PolynomialDecayLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

