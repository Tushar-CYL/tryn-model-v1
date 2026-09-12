"""Phase 1 wiring sanity — the classic overfit-one-batch test.

With every parameter trainable, the model must be able to memorise a single
batch (loss -> ~0). If it can't, the sequence assembly / masking is wrong.
"""
import torch

from common.decoder import DecoderConfig
from common.seed import set_seed
from image_model.data import make_dataset
from image_model.encoder import EncoderConfig
from image_model.model import ImageVLM


def test_overfit_single_batch():
    set_seed(0)
    ds = make_dataset(n_samples=8, n_classes=4, image_size=32, seed=0)
    images, text_ids = ds["images"], ds["text_ids"]

    enc = EncoderConfig(image_size=32, patch_size=8, d_model=64, depth=2)
    dec = DecoderConfig(vocab_size=ds["tokenizer"].vocab_size, d_model=64, depth=2)
    model = ImageVLM(enc, dec, connector_hidden=128)  # nothing frozen

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.loss(images, text_ids).detach())
    for _ in range(300):
        opt.zero_grad()
        loss = model.loss(images, text_ids)
        loss.backward()
        opt.step()
    last = float(loss.detach())

    assert last < first
    assert last < 0.1, f"expected to overfit one batch, got final loss {last:.3f}"
