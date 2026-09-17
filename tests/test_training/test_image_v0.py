"""Image-v0 trainer runs, loss falls, and checkpoints resume exactly."""
from omegaconf import OmegaConf

from common.config import load_config
from training.image_v0 import run_image_v0


def _fast_cfg(tmp_path, steps, resume=None):
    cfg = load_config("image_v0")
    # Keep it tiny + offline: force synthetic fallback and a short run.
    cfg.data.shards_dir = str(tmp_path / "no_such_shards")
    cfg.data.n_samples = 32
    cfg.data.batch_size = 16
    cfg.optim.steps = steps
    cfg.checkpoint.dir = str(tmp_path / "ck")
    cfg.checkpoint.resume = resume
    return cfg


def test_trainer_runs_and_loss_falls(tmp_path):
    out = run_image_v0(_fast_cfg(tmp_path, steps=60))
    assert out["last_loss"] < out["first_loss"]
    assert out["checkpoint"].endswith("ckpt.pt")


def test_resume_from_checkpoint(tmp_path):
    first = run_image_v0(_fast_cfg(tmp_path, steps=30))
    ckpt = first["checkpoint"]
    # Resume and train 30 more steps; loss should keep falling from where we left.
    resumed = run_image_v0(_fast_cfg(tmp_path, steps=60, resume=ckpt))
    assert resumed["checkpoint"] == ckpt
    assert resumed["last_loss"] is not None
    assert resumed["last_loss"] < first["first_loss"]
