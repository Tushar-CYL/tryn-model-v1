"""Config loading built on OmegaConf, with simple Hydra-style `defaults` support.

We keep this tiny on purpose: it resolves a `defaults:` list (a subset of Hydra's
composition) so configs can be loaded in plain scripts and tests without spinning
up the full Hydra runtime.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Repo-root/configs — resolved relative to this file (src/common/config.py).
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str, config_dir: Path | str | None = None) -> DictConfig:
    """Load a YAML config by name (with or without .yaml), resolving `defaults`.

    Only the two forms we use are supported in a `defaults:` list:
      - "base"          -> merge configs/base.yaml
      - "_self_"        -> merge this file's own keys (in list order)
    """
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR
    path = root / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = OmegaConf.load(path)
    defaults = raw.pop("defaults", None)
    if defaults is None:
        return raw  # type: ignore[return-value]

    merged = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            merged = OmegaConf.merge(merged, raw)
        else:
            merged = OmegaConf.merge(merged, load_config(str(entry), root))
    return merged  # type: ignore[return-value]
