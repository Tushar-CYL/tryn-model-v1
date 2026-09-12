"""Synthetic image-caption data for Phase 1 (no downloads).

Each class has a fixed caption and a fixed base image pattern; samples add noise.
A frozen encoder therefore maps each class to a stable feature, so a trainable
connector can learn to steer the frozen decoder toward the right caption — a
faithful miniature of Stage-2 alignment.
"""
from __future__ import annotations

import torch

from common.tokenizer import PAD, TinyTokenizer

# Captions use only the tokenizer's charset (a-z, 0-9, space, dot, comma).
_CAPTIONS = [
    "red circle",
    "blue square",
    "green star",
    "big house",
    "small tree",
    "old car",
    "new road",
    "long river",
]


def caption_for(class_id: int) -> str:
    return _CAPTIONS[class_id % len(_CAPTIONS)]


def _encode_padded(tok: TinyTokenizer, texts: list[str]) -> torch.Tensor:
    seqs = [tok.encode(t) for t in texts]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids


def make_dataset(
    n_samples: int,
    n_classes: int,
    image_size: int,
    in_channels: int = 3,
    seed: int = 0,
    noise: float = 0.3,
) -> dict:
    """Return {images, text_ids, classes, tokenizer}.

    images:   (N, C, H, W) float
    text_ids: (N, T) long   (BOS ... caption ... EOS, PAD-padded)
    classes:  (N,) long
    """
    n_classes = min(n_classes, len(_CAPTIONS))
    g = torch.Generator().manual_seed(seed)
    tok = TinyTokenizer()

    # Fixed per-class base image pattern.
    bases = torch.randn(n_classes, in_channels, image_size, image_size, generator=g)

    classes = torch.randint(0, n_classes, (n_samples,), generator=g)
    images = bases[classes] + noise * torch.randn(
        n_samples, in_channels, image_size, image_size, generator=g
    )
    text_ids = _encode_padded(tok, [caption_for(int(c)) for c in classes])
    return {"images": images, "text_ids": text_ids, "classes": classes, "tokenizer": tok}


def iter_batches(dataset: dict, batch_size: int, shuffle: bool = True, seed: int = 0):
    n = dataset["images"].size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle \
        else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {"images": dataset["images"][idx], "text_ids": dataset["text_ids"][idx]}
