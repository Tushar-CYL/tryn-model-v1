"""Phase 1 gate — connector-only alignment drives the loss down.

Freeze encoder + decoder, train only the connector on synthetic pairs, and
assert the loss falls meaningfully. Mirrors `training.align.run_alignment`.
"""
from common.config import load_config
from training.align import run_alignment


def test_connector_alignment_reduces_loss(tmp_path):
    cfg = load_config("align_tiny")
    cfg.output_dir = str(tmp_path)
    cfg.optim.steps = 200  # keep quick but enough to move
    result = run_alignment(cfg)

    assert result["last_loss"] < 0.9 * result["first_loss"], (
        f"connector failed to align: first={result['first_loss']:.3f} "
        f"last={result['last_loss']:.3f}"
    )
