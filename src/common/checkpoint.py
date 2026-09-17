"""Checkpoint save / load helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, **extra: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), **extra}
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload


def load_partial(path: str | Path, model: nn.Module, map_location: str = "cpu") -> dict:
    """Warm-start: copy only the checkpoint tensors whose name AND shape match.

    Safe across architecture changes (e.g. a differently-sized model) — mismatched
    or missing tensors are skipped rather than raising. Returns
    {loaded, skipped, total} counts.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    src = payload.get("model_state", payload)
    tgt = model.state_dict()
    compatible = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
    tgt.update(compatible)
    model.load_state_dict(tgt)
    return {"loaded": len(compatible), "skipped": len(src) - len(compatible),
            "total": len(tgt)}


def save_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer,
    step: int,
    **extra: Any,
) -> Path:
    """Checkpoint model + optimizer + step so a run can resume exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "step": step,
            **extra,
        },
        path,
    )
    return path


def load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    map_location: str = "cpu",
) -> dict:
    """Restore model (and optimizer, if given). Returns the payload incl. `step`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optim_state" in payload:
        optimizer.load_state_dict(payload["optim_state"])
    return payload
