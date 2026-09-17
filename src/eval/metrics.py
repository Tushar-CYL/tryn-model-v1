"""Simple metrics for the tiny image model."""
from __future__ import annotations

import torch

from common.tokenizer import BOS, EOS, PAD, UNK

_SPECIALS = frozenset({PAD, BOS, EOS, UNK})


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


def _strip_specials(ids: torch.Tensor) -> list[int]:
    """Caption content only: drop PAD/BOS/EOS/UNK (incl. trailing EOS padding
    that greedy generation appends to finished rows)."""
    return [int(t) for t in ids if int(t) not in _SPECIALS]


def exact_match(pred_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Fraction of rows whose caption content matches exactly."""
    n = pred_ids.size(0)
    hits = sum(
        _strip_specials(pred_ids[i]) == _strip_specials(target_ids[i]) for i in range(n)
    )
    return hits / max(n, 1)
