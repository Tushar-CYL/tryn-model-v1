"""Greedy caption generation from an ImageVLM."""
from __future__ import annotations

import torch

from common.tokenizer import BOS, EOS
from image_model.model import ImageVLM


@torch.no_grad()
def generate(
    model: ImageVLM, images: torch.Tensor, max_new_tokens: int = 20
) -> torch.Tensor:
    """Greedy decode. Returns generated token ids (B, L), excluding the BOS prompt."""
    model.eval()
    b = images.size(0)
    device = images.device
    text_ids = torch.full((b, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits = model.forward(images, text_ids)          # (B, Nv+T, V)
        next_tok = logits[:, -1, :].argmax(-1)            # (B,)
        next_tok = torch.where(finished, torch.full_like(next_tok, EOS), next_tok)
        text_ids = torch.cat([text_ids, next_tok[:, None]], dim=1)
        finished = finished | (next_tok == EOS)
        if bool(finished.all()):
            break
    return text_ids[:, 1:]  # drop BOS
