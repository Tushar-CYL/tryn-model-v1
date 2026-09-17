"""Image data pipeline: stream -> clean -> split -> shard (Phase 2, Week 3).

The pipeline is dataset-agnostic. Records flow through as a small dict:

    {"id": str, "image": PIL.Image, "caption": str}

Real sources stream image-caption pairs from the HuggingFace Hub; a synthetic
source produces the same record shape with no network, so the whole pipeline
(and its tests) runs offline on CPU. See `data/pipelines/DATASETS.md`.
"""
from __future__ import annotations

from .sources import Record, iter_source, list_datasets

__all__ = ["Record", "iter_source", "list_datasets"]
