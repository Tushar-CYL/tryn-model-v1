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


@torch.no_grad()
def generate_from_prompt(
    model: ImageVLM,
    image: torch.Tensor,
    prompt_ids: list[int],
    max_new_tokens: int = 12,
) -> list[int]:
    """Greedy-decode an answer for ONE image conditioned on a text prompt.

    `image` is (C, H, W) or (1, C, H, W); `prompt_ids` is the chat prefix
    (e.g. from `common.prompt.build_prompt_ids`). Returns the generated answer
    ids (prompt excluded, EOS trimmed).
    """
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    device = image.device
    text_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = text_ids.size(1)
    for _ in range(max_new_tokens):
        logits = model.forward(image, text_ids)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        text_ids = torch.cat([text_ids, next_tok], dim=1)
        if int(next_tok) == EOS:
            break
    gen = text_ids[0, prompt_len:].tolist()
    return [t for t in gen if t != EOS]
