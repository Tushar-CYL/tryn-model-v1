"""Phase 0 · Day 4 — toy loop trains, loss decreases, checkpoint saved."""
from pathlib import Path

from common.config import load_config
from training.toy_loop import run_toy_training


def test_toy_training_runs_and_saves(tmp_path):
    cfg = load_config("toy")
    cfg.output_dir = str(tmp_path)
    cfg.optim.steps = 100  # keep the test quick
    result = run_toy_training(cfg)

    assert result["last_loss"] < result["first_loss"]  # it learns
    assert Path(result["checkpoint"]).exists()          # checkpoint written
