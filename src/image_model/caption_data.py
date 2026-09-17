"""Caption data for the pretrained-LM VLM: images (224, SigLIP-normalized) +
HF-tokenized captions (Phase 2, proper).

Unlike the tiny char path, this tokenizes with the LM's own subword tokenizer.
Sequence per sample: [BOS] caption [EOS], right-padded. Labels score the caption
(+EOS); BOS and padding are masked (-100).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _load_images_and_captions(shards_dir, image_size, split, max_samples):
    from data_pipeline.shard import read_shards

    imgs, caps = [], []
    for rec in read_shards(Path(shards_dir) / split):
        im = rec.image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).float() / 255.0
        t = t.view(image_size, image_size, 3).permute(2, 0, 1).contiguous()
        imgs.append(t * 2.0 - 1.0)          # SigLIP expects [-1, 1]
        caps.append(rec.caption)
        if max_samples is not None and len(imgs) >= max_samples:
            break
    if not imgs:
        raise FileNotFoundError(f"No records under {shards_dir}/{split}")
    return torch.stack(imgs), caps


def load_caption_dataset(shards_dir, tokenizer, image_size=224, split="train",
                         max_len=48, max_samples=None) -> dict:
    """Return {images, input_ids, attention_mask, labels, captions}."""
    images, captions = _load_images_and_captions(shards_dir, image_size, split, max_samples)

    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    n = len(captions)
    input_ids = torch.full((n, max_len), pad, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    for i, cap in enumerate(captions):
        body = tokenizer(cap, add_special_tokens=False).input_ids
        ids = ([bos] + body + [eos])[:max_len]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
        lab = list(ids)
        lab[0] = -100                       # don't score predicting BOS
        labels[i, :L] = torch.tensor(lab, dtype=torch.long)
    return {"images": images, "input_ids": input_ids, "attention_mask": attn,
            "labels": labels, "captions": captions}


def iter_caption_batches(ds, batch_size, shuffle=True, seed=0):
    n = ds["images"].size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle \
        else torch.arange(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        yield {
            "images": ds["images"][idx],
            "input_ids": ds["input_ids"][idx],
            "attention_mask": ds["attention_mask"][idx],
            "labels": ds["labels"][idx],
            "captions": [ds["captions"][int(j)] for j in idx],
        }
