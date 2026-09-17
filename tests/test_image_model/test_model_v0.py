"""Week 4 assembly: from_config builds a tiny+resampler VLM that forwards,
overfits one batch (wiring correct), and supports chat-style answer supervision.
"""
import torch

from common.prompt import build_chat_example, collate_chat
from common.seed import set_seed
from common.tokenizer import TinyTokenizer
from image_model.data import make_dataset
from image_model.model import ImageVLM


def _cfg(vocab_size, num_latents=8):
    return {
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64,
                    "depth": 2, "n_heads": 4},
        "connector": {"type": "resampler", "num_latents": num_latents, "n_heads": 4,
                      "depth": 1},
        "decoder": {"vocab_size": vocab_size, "d_model": 64, "depth": 2, "max_seq_len": 128},
    }


def test_from_config_forward_uses_resampler_budget():
    tok = TinyTokenizer()
    model = ImageVLM.from_config(_cfg(tok.vocab_size, num_latents=8))
    assert model.num_vision_tokens == 8  # resampler fixes the budget

    b, t = 3, 7
    logits = model(torch.randn(b, 3, 32, 32), torch.randint(4, tok.vocab_size, (b, t)))
    assert logits.shape == (b, 8 + t, tok.vocab_size)


def test_overfit_one_batch_v0():
    set_seed(0)
    ds = make_dataset(n_samples=8, n_classes=4, image_size=32, seed=0)
    model = ImageVLM.from_config(_cfg(ds["tokenizer"].vocab_size, num_latents=8))

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.loss(ds["images"], ds["text_ids"]).detach())
    for _ in range(300):
        opt.zero_grad()
        loss = model.loss(ds["images"], ds["text_ids"])
        loss.backward()
        opt.step()
    last = float(loss.detach())
    assert last < first
    assert last < 0.1, f"v0 model failed to overfit one batch (loss {last:.3f})"


def test_chat_answer_supervision_runs():
    tok = TinyTokenizer()
    model = ImageVLM.from_config(_cfg(tok.vocab_size, num_latents=8))
    batch = collate_chat([
        build_chat_example(tok, "what color", "red"),
        build_chat_example(tok, "how many", "two"),
    ])
    images = torch.randn(2, 3, 32, 32)
    loss = model.loss(images, batch["input_ids"], text_labels=batch["labels"])
    assert loss.ndim == 0 and torch.isfinite(loss)
