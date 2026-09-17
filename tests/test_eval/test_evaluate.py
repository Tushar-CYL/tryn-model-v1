"""Held-out evaluation harness: metrics are well-formed and improve with training."""
import torch

from common.seed import set_seed
from eval.evaluate import evaluate
from image_model.data import make_dataset
from image_model.model import ImageVLM


def _cfg(vocab):
    return {
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2},
        "connector": {"type": "resampler", "num_latents": 8, "depth": 1},
        "decoder": {"vocab_size": vocab, "d_model": 64, "depth": 2, "max_seq_len": 128},
    }


def test_eval_report_is_well_formed():
    set_seed(0)
    ds = make_dataset(n_samples=16, n_classes=4, image_size=32, seed=0)
    model = ImageVLM.from_config(_cfg(ds["tokenizer"].vocab_size))
    rep = evaluate(model, ds, batch_size=8)
    assert torch.isfinite(torch.tensor(rep.loss))
    assert 0.0 <= rep.token_acc <= 1.0
    assert 0.0 <= rep.exact_match <= 1.0
    assert rep.n == 16 and len(rep.samples) > 0
    assert set(rep.samples[0]) == {"target", "generated"}


def test_training_improves_eval_metrics():
    set_seed(0)
    ds = make_dataset(n_samples=8, n_classes=4, image_size=32, seed=0)
    model = ImageVLM.from_config(_cfg(ds["tokenizer"].vocab_size))
    before = evaluate(model, ds, batch_size=8)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        model.loss(ds["images"], ds["text_ids"]).backward()
        opt.step()
    after = evaluate(model, ds, batch_size=8)

    assert after.token_acc > before.token_acc
    assert after.exact_match >= before.exact_match
    assert after.exact_match > 0.5  # memorised captions are recoverable by greedy decode
