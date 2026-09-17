"""Synthetic image-caption data for Phase 1 (no downloads).

Each class has a fixed caption and a fixed base image pattern; samples add noise.
A frozen encoder therefore maps each class to a stable feature, so a trainable
connector can learn to steer the frozen decoder toward the right caption — a
faithful miniature of Stage-2 alignment.
"""
from __future__ import annotations

from pathlib import Path

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
    base_seed: int = 0,
) -> dict:
    """Return {images, text_ids, classes, tokenizer}.

    images:   (N, C, H, W) float
    text_ids: (N, T) long   (BOS ... caption ... EOS, PAD-padded)
    classes:  (N,) long

    Per-class base patterns are keyed by `base_seed`; `seed` varies the samples.
    Hold `base_seed` fixed across train/held-out so the class->image mapping is
    shared (a real generalisation test) and only the noise differs.
    """
    n_classes = min(n_classes, len(_CAPTIONS))
    g = torch.Generator().manual_seed(seed)
    tok = TinyTokenizer()

    # Fixed per-class base image pattern (independent of the sampling seed).
    bg = torch.Generator().manual_seed(base_seed)
    bases = torch.randn(n_classes, in_channels, image_size, image_size, generator=bg)

    classes = torch.randint(0, n_classes, (n_samples,), generator=g)
    images = bases[classes] + noise * torch.randn(
        n_samples, in_channels, image_size, image_size, generator=g
    )
    text_ids = _encode_padded(tok, [caption_for(int(c)) for c in classes])
    return {"images": images, "text_ids": text_ids, "classes": classes, "tokenizer": tok}


def load_shard_dataset(
    shards_dir: str | Path,
    image_size: int,
    split: str = "train",
    max_samples: int | None = None,
    tokenizer: TinyTokenizer | None = None,
    max_text_len: int | None = None,
    normalize: str | None = None,
) -> dict:
    """Load real image-caption shards (built by `data_pipeline.build`) into the
    same {images, text_ids, tokenizer} dict shape as `make_dataset`.

    Images are bicubic-resized to `image_size`. Pixel scaling depends on the
    encoder: `normalize=None` gives [0, 1] (tiny encoder); `normalize="siglip"`
    gives [-1, 1] (SigLIP's expected input). Captions are encoded with the tiny
    tokenizer (out-of-charset chars map to <unk>). `max_text_len` caps the token
    sequence (incl. BOS/EOS) so it fits the decoder's budget.
    """
    from PIL import Image

    from data_pipeline.shard import read_shards  # local import: optional dep path

    tok = tokenizer or TinyTokenizer()
    images, captions = [], []
    for rec in read_shards(Path(shards_dir) / split):
        img = rec.image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        arr = bytearray(img.tobytes())  # writable buffer (avoids frombuffer warning)
        t = torch.frombuffer(arr, dtype=torch.uint8).float() / 255.0
        t = t.view(image_size, image_size, 3).permute(2, 0, 1).contiguous()
        if normalize == "siglip":
            t = t * 2.0 - 1.0  # (x/255 - 0.5) / 0.5  ->  [-1, 1]
        elif normalize not in (None, "unit"):
            raise ValueError(f"Unknown normalize mode: {normalize!r}")
        images.append(t)
        captions.append(rec.caption)
        if max_samples is not None and len(images) >= max_samples:
            break

    if not images:
        raise FileNotFoundError(f"No records found under {shards_dir}/{split}")
    text_ids = _encode_padded(tok, captions)
    if max_text_len is not None and text_ids.size(1) > max_text_len:
        text_ids = _truncate_keep_eos(text_ids, max_text_len)
    return {
        "images": torch.stack(images),
        "text_ids": text_ids,
        "tokenizer": tok,
    }


def _truncate_keep_eos(text_ids: torch.Tensor, max_len: int) -> torch.Tensor:
    """Truncate each row to `max_len`, forcing the last real token to EOS."""
    from common.tokenizer import EOS

    out = text_ids[:, :max_len].clone()
    out[:, -1] = EOS
    return out


def iter_batches(dataset: dict, batch_size: int, shuffle: bool = True, seed: int = 0):
    n = dataset["images"].size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle \
        else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {"images": dataset["images"][idx], "text_ids": dataset["text_ids"][idx]}
