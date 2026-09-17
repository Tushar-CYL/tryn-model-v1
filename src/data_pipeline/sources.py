"""Data sources: HuggingFace streaming + an offline synthetic source.

Every source yields a uniform `Record` so the downstream clean/split/shard
stages don't care where the data came from. Real datasets are described in a
small registry (`DATASETS`) that maps a short name to its Hub id, split, and
the columns holding the image and caption.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image


@dataclass
class Record:
    """One image-caption training example flowing through the pipeline."""

    id: str
    image: Image.Image
    caption: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """How to stream one real dataset from the Hub."""

    hub_id: str
    split: str
    image_col: str
    caption_col: str
    config: str | None = None
    note: str = ""


# Registry of the image datasets named in the master plan (§8) / week-3 plan.
# `image_col` / `caption_col` are the columns to read; some sets nest captions
# as a list, which `_extract_caption` flattens.
DATASETS: dict[str, DatasetSpec] = {
    "coco": DatasetSpec(
        hub_id="clip-benchmark/wds_mscoco_captions",
        split="test",
        image_col="jpg",
        caption_col="txt",
        note="COCO 2014 captions as WebDataset (inline jpg + txt). Public, streams well.",
    ),
    "coco_karpathy": DatasetSpec(
        hub_id="yerevann/coco-karpathy",
        split="train",
        image_col="url",
        caption_col="sentences",
        note="Karpathy COCO split; image referenced by URL (needs a fetch step).",
    ),
    "cc3m": DatasetSpec(
        hub_id="pixparse/cc3m-wds",
        split="train",
        image_col="jpg",
        caption_col="txt",
        note="Conceptual Captions 3M (WebDataset). Large; use --limit.",
    ),
}


def list_datasets() -> dict[str, DatasetSpec]:
    """Return the dataset registry (name -> spec)."""
    return dict(DATASETS)


def _stable_id(prefix: str, i: int, caption: str) -> str:
    h = hashlib.sha1(caption.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{i:07d}-{h}"


def _extract_caption(value) -> str:
    """Captions come as a str, a list of str, or a list of dicts. Take the first.

    WebDataset COCO packs all 5 reference captions into one newline-separated
    string; we keep just the first line so a sample is one image + one caption.
    """
    if isinstance(value, str):
        return value.splitlines()[0].strip() if value.strip() else value
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, dict):  # e.g. {"raw": "..."} style rows
            for k in ("raw", "caption", "text", "sentence"):
                if k in first:
                    return str(first[k])
            return str(first)
        return str(first)
    return str(value)


def iter_hf(
    name: str,
    limit: int | None = None,
    *,
    streaming: bool = True,
) -> Iterator[Record]:
    """Stream a registered dataset from the HuggingFace Hub.

    Imports `datasets` lazily so the offline synthetic path never requires it.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    spec = DATASETS[name]

    from datasets import load_dataset  # lazy: only needed for real pulls

    ds = load_dataset(spec.hub_id, spec.config, split=spec.split, streaming=streaming)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        image = row.get(spec.image_col)
        if not isinstance(image, Image.Image):
            # Some sets store bytes/URLs; skip anything we can't treat as an image
            # here (the real fetch/decode belongs in a dedicated loader). Keep the
            # pipeline honest rather than silently fabricating pixels.
            continue
        caption = _extract_caption(row.get(spec.caption_col))
        yield Record(
            id=_stable_id(name, i, caption),
            image=image.convert("RGB"),
            caption=caption,
            meta={"source": name},
        )


def iter_synthetic(
    n: int,
    *,
    image_size: int = 32,
    seed: int = 0,
    inject_bad: bool = False,
) -> Iterator[Record]:
    """Offline stand-in source with the same Record shape as `iter_hf`.

    Produces `n` deterministic image-caption pairs drawn from a small fixed set
    of scenes. With `inject_bad=True` it also emits a few dirty records
    (duplicate captions, a tiny/broken image, a non-English caption) so the
    cleaning stage has something to remove in tests.
    """
    import random

    rng = random.Random(seed)
    scenes = [
        ("a red circle on a white wall", (200, 60, 60)),
        ("a blue square on the floor", (60, 60, 200)),
        ("a green star in the sky", (60, 180, 60)),
        ("a small brown house by a road", (150, 110, 70)),
    ]
    for i in range(n):
        caption, color = scenes[i % len(scenes)]
        jitter = tuple(min(255, max(0, c + rng.randint(-20, 20))) for c in color)
        img = Image.new("RGB", (image_size, image_size), jitter)
        yield Record(
            id=_stable_id("syn", i, caption + str(i)),
            image=img,
            caption=caption,
            meta={"source": "synthetic", "scene": i % len(scenes)},
        )

    if inject_bad:
        # duplicate caption (same text as first scene) -> dedup should drop it
        yield Record("syn-dup-0", Image.new("RGB", (image_size, image_size), (200, 60, 60)),
                     scenes[0][0], {"source": "synthetic", "dirty": "dup"})
        # tiny image -> min-size filter should drop it
        yield Record("syn-tiny-0", Image.new("RGB", (4, 4), (0, 0, 0)),
                     "a tiny thing", {"source": "synthetic", "dirty": "tiny"})
        # non-English caption -> language filter should drop it
        yield Record("syn-lang-0", Image.new("RGB", (image_size, image_size), (10, 10, 10)),
                     "これは日本語のキャプション",
                     {"source": "synthetic", "dirty": "lang"})
        # empty caption -> length filter should drop it
        yield Record("syn-empty-0", Image.new("RGB", (image_size, image_size), (20, 20, 20)),
                     "  ", {"source": "synthetic", "dirty": "empty"})


def iter_source(
    name: str,
    limit: int | None = None,
    *,
    seed: int = 0,
    inject_bad: bool = False,
) -> Iterator[Record]:
    """Unified entry point. ``name='synthetic'`` runs offline; anything else is
    looked up in the Hub registry and streamed."""
    if name == "synthetic":
        yield from iter_synthetic(limit or 32, seed=seed, inject_bad=inject_bad)
    else:
        yield from iter_hf(name, limit=limit)
