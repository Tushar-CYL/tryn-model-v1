"""A tiny from-scratch ViT-style patch encoder (random init, no downloads).

Stands in for SigLIP/ViT during Phase 0/1. Same interface a real encoder would
expose: image (B, C, H, W) -> patch features (B, num_patches, d_model). In Phase 2
this is swapped for an open SigLIP init; the connector interface stays identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncoderConfig:
    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    d_model: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


class _EncBlock(nn.Module):
    """Bidirectional transformer block (no causal mask)."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyPatchEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.cfg = cfg
        self.patch = nn.Conv2d(
            cfg.in_channels, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.d_model))
        self.blocks = nn.ModuleList([_EncBlock(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def num_patches(self) -> int:
        return self.cfg.num_patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, d_model, H/p, W/p) -> (B, num_patches, d_model)
        x = self.patch(images).flatten(2).transpose(1, 2)
        x = x + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)
