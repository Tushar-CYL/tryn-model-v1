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

    @property
    def out_tokens(self) -> int | None:
        """Token count out == token count in (MLP is per-token)."""
        return None


class PerceiverResampler(nn.Module):
    """Compress a variable number of encoder features to a fixed token budget.

    A small set of learned latent queries cross-attend to the encoder features,
    so `N` patch/frame tokens (196+ for real SigLIP) become `num_latents`
    decoder tokens — the key lever for the video token explosion later, and a
    cheaper prompt for image too.

    Input:  (B, N, in_dim)
    Output: (B, num_latents, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_latents: int = 16,
        n_heads: int = 4,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, out_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(out_dim),
                        "ln_kv": nn.LayerNorm(out_dim),
                        "attn": nn.MultiheadAttention(out_dim, n_heads, batch_first=True),
                        "ln_ff": nn.LayerNorm(out_dim),
                        "ff": nn.Sequential(
                            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
                            nn.Linear(out_dim * 2, out_dim),
                        ),
                    }
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        kv = self.in_proj(x)                       # (B, N, out_dim)
        q = self.latents.expand(b, -1, -1)         # (B, L, out_dim)
        for layer in self.layers:
            qn = layer["ln_q"](q)
            kvn = layer["ln_kv"](kv)
            a, _ = layer["attn"](qn, kvn, kvn, need_weights=False)
            q = q + a
            q = q + layer["ff"](layer["ln_ff"](q))
        return q

    @property
    def out_tokens(self) -> int:
        return self.num_latents


def build_connector(cfg) -> nn.Module:
    """Factory: build the connector named by `cfg.type` ('mlp' | 'resampler').

    `cfg` is a mapping/DictConfig with at least `type`, `in_dim`, `out_dim`.
    """
    kind = cfg.get("type", "mlp")
    if kind == "mlp":
        return MLPConnector(cfg["in_dim"], cfg["out_dim"], cfg.get("hidden_dim"))
    if kind == "resampler":
        return PerceiverResampler(
            cfg["in_dim"],
            cfg["out_dim"],
            num_latents=cfg.get("num_latents", 16),
            n_heads=cfg.get("n_heads", 4),
            depth=cfg.get("depth", 1),
        )
    raise ValueError(f"Unknown connector type: {kind!r}")
