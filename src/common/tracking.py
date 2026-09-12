"""W&B tracking helper.

Network-free by default: the base config uses `tracking.mode: disabled`, so runs
on the laptop / in CI never block on a login. Flip to `online` on the GPU box.
Falls back to a no-op run if wandb is missing or init fails, so training code can
always call `.log(...)` / `.finish()` unconditionally.
"""
from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .logging_utils import get_logger

log = get_logger(__name__)


class _NoOpRun:
    """Stand-in for a wandb run when tracking is off or unavailable."""

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:  # noqa: D401
        return None

    def finish(self) -> None:
        return None


def init_tracking(cfg: DictConfig) -> Any:
    """Initialise experiment tracking from the `tracking` config block."""
    tcfg = cfg.get("tracking", {})
    backend = tcfg.get("backend", "none")
    mode = tcfg.get("mode", "disabled")

    if backend != "wandb" or mode == "disabled":
        log.info("Tracking disabled (backend=%s, mode=%s) -> no-op run.", backend, mode)
        return _NoOpRun()

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("wandb import failed (%s); using no-op run.", exc)
        return _NoOpRun()

    try:
        run = wandb.init(
            project=tcfg.get("project", "perception-slm"),
            entity=tcfg.get("entity"),
            name=tcfg.get("run_name"),
            mode=mode,  # "online" | "offline"
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        log.info("wandb run started (mode=%s).", mode)
        return run
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("wandb.init failed (%s); using no-op run.", exc)
        return _NoOpRun()
