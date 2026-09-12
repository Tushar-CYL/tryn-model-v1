"""Phase 1 — assembled ImageVLM forward pass produces correct shapes + scalar loss."""
import torch

from common.decoder import DecoderConfig
from common.tokenizer import TinyTokenizer
from image_model.encoder import EncoderConfig
from image_model.model import ImageVLM


def _model():
    tok = TinyTokenizer()
    enc = EncoderConfig(image_size=32, patch_size=8, in_channels=3, d_model=64, depth=2)
    dec = DecoderConfig(vocab_size=tok.vocab_size, d_model=64, depth=2, max_seq_len=128)
    return ImageVLM(enc, dec, connector_hidden=128), tok


def test_forward_and_loss_shapes():
    model, tok = _model()
    b, t = 4, 9
    images = torch.randn(b, 3, 32, 32)
    text_ids = torch.randint(4, tok.vocab_size, (b, t))
    logits = model(images, text_ids)
    nv = model.num_vision_tokens  # (32/8)^2 = 16
    assert nv == 16
    assert logits.shape == (b, nv + t, tok.vocab_size)

    loss = model.loss(images, text_ids)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_vision_tokens_shape():
    model, _ = _model()
    v = model.vision_tokens(torch.randn(2, 3, 32, 32))
    assert v.shape == (2, 16, 64)
