"""A tiny from-scratch decoder-only transformer (the "SLM").

Small enough to train on CPU in seconds. Accepts either token ids or
pre-computed input embeddings, so a connector can prepend vision tokens
(sequence assembly `[vision | text]`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    max_seq_len: int = 128


class _Block(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> embeddings, so a connector can concatenate vision tokens."""
        return self.tok_emb(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B, T, vocab_size). Provide ids OR embeds."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids / inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        b, t, _ = inputs_embeds.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} > max_seq_len {self.cfg.max_seq_len}.")

        x = inputs_embeds + self.pos_emb[:, :t]
        # Causal mask: True = disallow attending (nn.MultiheadAttention convention).
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
        )
        for blk in self.blocks:
            x = blk(x, causal)
        x = self.ln_f(x)
        return self.lm_head(x)


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy. Positions with label == -100 are ignored.

    Shifts so token t predicts token t+1.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
