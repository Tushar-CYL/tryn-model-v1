"""Minimal LoRA for `nn.Linear` (Phase 2, Week 6 — Stage-3 instruction tuning).

Low-Rank Adaptation freezes the base weight `W` and learns a small update
`B @ A` (rank `r`), so instruction tuning updates a few thousand params instead
of the whole decoder. This is a compact, dependency-free implementation (the
production path would use `peft`); the math and the freeze/merge semantics match.

    y = W x + (alpha / r) * B(A(x))
"""
from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze W (and bias)

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Standard init: A ~ small normal, B = 0 so the adapter starts as identity.
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter back into a plain `nn.Linear` (for export/serving)."""
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)  # (out, in)
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


# NB: don't target `out_proj` inside nn.MultiheadAttention — it reads
# `out_proj.weight` directly, so wrapping it breaks attention. LoRA the MLP
# blocks and the LM head, which are used through a normal forward().
_DEFAULT_TARGETS = ("mlp", "lm_head")


def apply_lora(
    root: nn.Module,
    r: int = 4,
    alpha: int = 8,
    targets: tuple[str, ...] = _DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> int:
    """Replace matching `nn.Linear` submodules of `root` with `LoRALinear`.

    A module matches if any string in `targets` occurs in its dotted name.
    Returns the number of layers adapted.
    """
    to_replace = [
        (name, m) for name, m in root.named_modules()
        if isinstance(m, nn.Linear) and any(t in name for t in targets)
    ]
    for name, module in to_replace:
        parent = root
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
    return len(to_replace)


def lora_parameters(root: nn.Module):
    """Iterate the trainable LoRA parameters under `root`."""
    for m in root.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def merge_lora(root: nn.Module) -> int:
    """Merge every `LoRALinear` under `root` back into plain `nn.Linear`. Returns count."""
    merged = [(n, m) for n, m in root.named_modules() if isinstance(m, LoRALinear)]
    for name, module in merged:
        parent = root
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, attr, module.merge())
    return len(merged)
