"""Cleaning / filtering stage (Week 3, Day 3).

Drops records that would only add noise to alignment training:

  * broken / too-small images
  * empty or too-short/long captions
  * non-English captions (cheap ASCII-ratio heuristic — no model download)
  * exact duplicates (by caption text and by image content hash)
  * optional CLIP-score filter (image/caption agreement), only if a CLIP model
    is provided — off by default so the pipeline stays download-free.

Every stage increments a counter so the run can log kept/dropped stats.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .sources import Record


@dataclass
class CleanConfig:
    min_size: int = 16           # min image side in px
    min_caption_chars: int = 5
    max_caption_chars: int = 300
    min_ascii_ratio: float = 0.8  # English-ish heuristic
    dedup_captions: bool = True
    dedup_images: bool = True
    clip_min_score: float | None = None  # e.g. 0.20; requires clip_scorer


@dataclass
class CleanStats:
    seen: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {"seen": self.seen, "kept": self.kept, "dropped": dict(self.dropped)}


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isascii() for c in text) / len(text)


def _image_hash(img) -> str:
    """Content hash of a downscaled grayscale image (near-dup catcher)."""
    small = img.convert("L").resize((16, 16))
    return hashlib.sha1(small.tobytes()).hexdigest()


def clean(
    records: Iterable[Record],
    cfg: CleanConfig | None = None,
    *,
    clip_scorer: Callable[[Record], float] | None = None,
) -> tuple[Iterator[Record], CleanStats]:
    """Filter `records`, returning a generator of survivors and a live stats obj.

    Stats are populated as the generator is consumed, so read them *after*
    iterating (or after `list(...)`).
    """
    cfg = cfg or CleanConfig()
    stats = CleanStats()
    seen_captions: set[str] = set()
    seen_images: set[str] = set()

    def _gen() -> Iterator[Record]:
        for rec in records:
            stats.seen += 1

            caption = (rec.caption or "").strip()
            if len(caption) < cfg.min_caption_chars:
                stats.dropped["caption_too_short"] += 1
                continue
            if len(caption) > cfg.max_caption_chars:
                stats.dropped["caption_too_long"] += 1
                continue
            if _ascii_ratio(caption) < cfg.min_ascii_ratio:
                stats.dropped["non_english"] += 1
                continue

            try:
                w, h = rec.image.size
            except Exception:
                stats.dropped["broken_image"] += 1
                continue
            if min(w, h) < cfg.min_size:
                stats.dropped["image_too_small"] += 1
                continue

            if cfg.dedup_captions:
                key = caption.lower()
                if key in seen_captions:
                    stats.dropped["dup_caption"] += 1
                    continue
                seen_captions.add(key)

            if cfg.dedup_images:
                try:
                    ih = _image_hash(rec.image)
                except Exception:
                    stats.dropped["broken_image"] += 1
                    continue
                if ih in seen_images:
                    stats.dropped["dup_image"] += 1
                    continue
                seen_images.add(ih)

            if cfg.clip_min_score is not None and clip_scorer is not None:
                if clip_scorer(rec) < cfg.clip_min_score:
                    stats.dropped["low_clip_score"] += 1
                    continue

            rec.caption = caption
            stats.kept += 1
            yield rec

    return _gen(), stats
