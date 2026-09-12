"""Phase 0 · Day 5 — a deliberately trivial "fake VLM".

Random image tensor -> linear -> token logits. Not a real model: it exists to
prove the harness (forward + one train step + eval + checkpoint) end-to-end
before any real modelling. See `training/toy_loop.py` for the MLP counterpart.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FakeVLM(nn.Module):
    def __init__(self, image_size: int, in_channels: int, seq_len: int, vocab_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        in_features = in_channels * image_size * image_size
        self.proj = nn.Linear(in_features, seq_len * vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        flat = images.reshape(b, -1)
        return self.proj(flat).view(b, self.seq_len, self.vocab_size)

    def loss(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target_ids.reshape(-1)
        )


def train_step(model: FakeVLM, images: torch.Tensor, target_ids: torch.Tensor, opt) -> float:
    model.train()
    opt.zero_grad()
    loss = model.loss(images, target_ids)
    loss.backward()
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model: FakeVLM, images: torch.Tensor, target_ids: torch.Tensor) -> dict:
    model.eval()
    logits = model.forward(images)
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
    acc = (logits.argmax(-1) == target_ids).float().mean()
    return {"loss": float(loss), "acc": float(acc)}
