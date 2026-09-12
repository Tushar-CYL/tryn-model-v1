"""The connector: projects encoder features into the decoder's token space.

This is the piece that is genuinely "from scratch" and trained on our
(modality, text) pairs — the defensible asset in the master plan. v0 is the
simple LLaVA-style MLP projector that keeps all tokens.
"""
from __future__ import annotations

import torch
from torch import nn


class MLPConnector(nn.Module):
    """Two-layer MLP mapping encoder features -> decoder embedding space.

    Input:  (B, N, in_dim)  encoder patch/frame features
    Output: (B, N, out_dim) tokens in the decoder's embedding space
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
