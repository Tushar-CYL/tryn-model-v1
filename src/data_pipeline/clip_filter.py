"""Optional CLIP-score filter for caption quality (Week 3, Day 3 — real version).

Scores image/caption agreement with an open CLIP model and lets the cleaning
stage drop pairs below a threshold (mismatched or junk captions). Off by default
because it downloads a CLIP model; enable with `--clip-min-score` in the builder.
"""
from __future__ import annotations

from typing import Callable

from .sources import Record


def build_clip_scorer(model_name: str = "openai/clip-vit-base-patch32",
                      device: str = "cpu") -> Callable[[Record], float]:
    """Return a function Record -> cosine similarity (image, caption) in [-1, 1].

    Lazily imports transformers/torch so the offline pipeline never needs them.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(rec: Record) -> float:
        inputs = proc(text=[rec.caption], images=[rec.image], return_tensors="pt",
                      padding=True, truncation=True).to(device)
        out = model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img * txt).sum(-1)[0])

    return score
