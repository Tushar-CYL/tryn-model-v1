"""Freeze a held-out eval split + a small golden set (Week 3, Day 4).

The split is decided by hashing each record id into [0, 1), so it is:
  * deterministic — the same id always lands in the same split, across runs;
  * stable under growth — adding data never reshuffles existing assignments;
  * leak-proof — a single source of truth (`assign_split`) is used by both the
    trainer and the evaluator.

`freeze_split` writes a versioned manifest (the sorted eval/golden ids + a
sha256 over them) so we can prove, later, that training never touched them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Split = Literal["train", "val", "golden"]


@dataclass(frozen=True)
class SplitConfig:
    val_frac: float = 0.10
    golden_frac: float = 0.02
    salt: str = "perception-slm-v0"


def _unit_hash(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def assign_split(rec_id: str, cfg: SplitConfig | None = None) -> Split:
    """Map a record id to its split. The one source of truth for both sides."""
    cfg = cfg or SplitConfig()
    u = _unit_hash(rec_id, cfg.salt)
    if u < cfg.golden_frac:
        return "golden"
    if u < cfg.golden_frac + cfg.val_frac:
        return "val"
    return "train"


def freeze_split(
    ids: Iterable[str],
    out_path: str | Path,
    cfg: SplitConfig | None = None,
) -> dict:
    """Write a frozen manifest of the eval (val + golden) ids with a checksum."""
    cfg = cfg or SplitConfig()
    val, golden = [], []
    n_train = 0
    for rec_id in ids:
        s = assign_split(rec_id, cfg)
        if s == "val":
            val.append(rec_id)
        elif s == "golden":
            golden.append(rec_id)
        else:
            n_train += 1
    val.sort()
    golden.sort()

    checksum = hashlib.sha256(
        json.dumps({"val": val, "golden": golden}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    manifest = {
        "version": cfg.salt,
        "config": {"val_frac": cfg.val_frac, "golden_frac": cfg.golden_frac},
        "counts": {"train": n_train, "val": len(val), "golden": len(golden)},
        "checksum": checksum,
        "val_ids": val,
        "golden_ids": golden,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class FrozenEvalSet:
    """Loaded eval manifest with a fast membership guard against leakage."""

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        self._eval_ids = set(manifest["val_ids"]) | set(manifest["golden_ids"])

    @classmethod
    def load(cls, path: str | Path) -> "FrozenEvalSet":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def is_eval(self, rec_id: str) -> bool:
        return rec_id in self._eval_ids

    def assert_not_training(self, rec_id: str) -> None:
        if rec_id in self._eval_ids:
            raise ValueError(f"Leakage: eval id {rec_id!r} entered the training stream.")

    def verify_checksum(self) -> bool:
        recomputed = hashlib.sha256(
            json.dumps(
                {"val": self.manifest["val_ids"], "golden": self.manifest["golden_ids"]},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return recomputed == self.manifest["checksum"]
