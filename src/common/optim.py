"""Optimizer + LR schedule helpers (shared training quality).

- `build_optimizer`: AdamW with weight decay applied to matmul weights only
  (biases and LayerNorm/embedding params are excluded, the standard recipe).
- `cosine_warmup`: linear warmup then cosine decay to a floor. Returned as a
  plain function step -> lr-multiplier so it works with any optimizer without
  extra dependencies.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """AdamW with no weight decay on biases / norm / 1-D params."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def cosine_warmup(total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> Callable[[int], float]:
    """Return f(step) -> lr multiplier in [min_ratio, 1]. Linear warmup + cosine."""
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr
