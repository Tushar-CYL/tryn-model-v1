"""Phase 0 · Day 5 — fake VLM does forward + one train step + eval end-to-end."""
import torch

from image_model.fake_vlm import FakeVLM, evaluate, train_step


def _batch(b=8, c=3, hw=16, seq_len=4, vocab=20):
    g = torch.Generator().manual_seed(0)
    images = torch.randn(b, c, hw, hw, generator=g)
    targets = torch.randint(0, vocab, (b, seq_len), generator=g)
    return images, targets


def test_forward_shape():
    model = FakeVLM(image_size=16, in_channels=3, seq_len=4, vocab_size=20)
    images, _ = _batch()
    logits = model(images)
    assert logits.shape == (8, 4, 20)


def test_train_step_and_eval():
    model = FakeVLM(image_size=16, in_channels=3, seq_len=4, vocab_size=20)
    images, targets = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = evaluate(model, images, targets)
    for _ in range(50):
        train_step(model, images, targets, opt)
    after = evaluate(model, images, targets)

    assert after["loss"] < before["loss"]
    assert 0.0 <= after["acc"] <= 1.0
