"""Simple metrics for the tiny image model."""
from __future__ import annotations

import torch

from common.tokenizer import PAD


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Next-token accuracy over non-ignored positions (label != -100)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (shift_logits.argmax(-1) == shift_labels) & mask
    return float(correct.sum() / mask.sum())


def exact_match(pred_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Fraction of rows where the (PAD/special-stripped) sequences match exactly."""
    n = pred_ids.size(0)
    hits = 0
    for i in range(n):
        p = [int(t) for t in pred_ids[i] if int(t) != PAD]
        # target rows are BOS ... caption ... EOS (PAD-padded); strip PAD only,
        # generation already excludes BOS.
        t = [int(t) for t in target_ids[i] if int(t) != PAD]
        if t and t[0] == 1:  # BOS
            t = t[1:]
        if p == t:
            hits += 1
    return hits / max(n, 1)
