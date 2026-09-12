"""Phase 0 · Day 4 — minimal training harness.

Trains a 2-layer MLP on random tensors, logs to W&B (no-op unless enabled),
and saves a checkpoint. This proves the plumbing, not a model.

Run:  python -m training.toy_loop            # uses configs/toy.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from common.checkpoint import save_checkpoint
from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking

log = get_logger(__name__)


class ToyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_toy_training(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    g = torch.Generator().manual_seed(cfg.seed)
    x = torch.randn(cfg.data.n_samples, cfg.model.in_dim, generator=g)
    # Fixed random linear target -> a learnable regression problem.
    w = torch.randn(cfg.model.in_dim, cfg.model.out_dim, generator=g)
    y = x @ w

    model = ToyMLP(cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    loss_fn = nn.MSELoss()

    x, y = x.to(device), y.to(device)
    n, bs = cfg.data.n_samples, cfg.data.batch_size
    first_loss = last_loss = None
    for step in range(cfg.optim.steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if first_loss is None:
            first_loss = last_loss
        if step % 50 == 0:
            log.info("step %d | loss %.4f", step, last_loss)
        run.log({"train/loss": last_loss}, step=step)

    ckpt = save_checkpoint(
        Path(cfg.output_dir) / cfg.experiment / "toy_mlp.pt",
        model,
        step=cfg.optim.steps,
        final_loss=last_loss,
    )
    run.finish()
    log.info("first_loss=%.4f last_loss=%.4f ckpt=%s", first_loss, last_loss, ckpt)
    return {"first_loss": first_loss, "last_loss": last_loss, "checkpoint": str(ckpt)}


def main() -> None:
    run_toy_training(load_config("toy"))


if __name__ == "__main__":
    main()
