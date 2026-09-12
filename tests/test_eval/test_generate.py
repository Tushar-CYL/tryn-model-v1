"""Eval — greedy generation runs and metrics compute over valid ranges."""
import torch

from common.decoder import DecoderConfig, lm_loss
from common.tokenizer import TinyTokenizer
from eval.generate import generate
from eval.metrics import token_accuracy
from image_model.encoder import EncoderConfig
from image_model.model import ImageVLM


def _model():
    tok = TinyTokenizer()
    enc = EncoderConfig(image_size=32, patch_size=8, d_model=64, depth=2)
    dec = DecoderConfig(vocab_size=tok.vocab_size, d_model=64, depth=2, max_seq_len=128)
    return ImageVLM(enc, dec, connector_hidden=128), tok


def test_generate_shapes():
    model, _ = _model()
    images = torch.randn(3, 3, 32, 32)
    out = generate(model, images, max_new_tokens=10)
    assert out.shape[0] == 3
    assert out.shape[1] <= 10


def test_token_accuracy_range():
    model, tok = _model()
    images = torch.randn(2, 3, 32, 32)
    text_ids = torch.randint(4, tok.vocab_size, (2, 8))
    logits = model(images, text_ids)
    nv = model.num_vision_tokens
    labels = torch.full((2, nv + 8), -100)
    labels[:, nv:] = text_ids
    acc = token_accuracy(logits, labels)
    assert 0.0 <= acc <= 1.0
    # sanity: lm_loss on the same tensors is a finite scalar
    assert torch.isfinite(lm_loss(logits, labels))
