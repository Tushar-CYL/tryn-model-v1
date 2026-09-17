"""Stage-3 LoRA instruction tuning learns VQA and beats the majority baseline."""
from common.config import load_config
from training.instruct import run_instruct


def test_instruct_beats_baseline(tmp_path):
    cfg = load_config("instruct")
    cfg.optim.steps = 400
    cfg.data.n_samples = 256
    cfg.checkpoint.dir = str(tmp_path / "ck")

    out = run_instruct(cfg)
    assert out["last_loss"] < out["first_loss"]
    assert out["vqa_after"] > out["vqa_before"]
    assert out["beats_baseline"] is True
    assert out["vqa_after"] >= 0.8  # held-out generalisation to unseen samples
