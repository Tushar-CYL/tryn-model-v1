"""LM-VLM plumbing with a tiny random Llama (no download): forward/loss/generate/save."""
import torch

from common.connector import build_connector
from image_model.encoder import build_encoder
from image_model.vlm_lm import LMImageVLM


class _FakeTok:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in seq) for seq in seqs]


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
                      max_position_embeddings=128)
    lm = LlamaForCausalLM(cfg)
    enc = build_encoder({"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2})
    conn = build_connector({"type": "resampler", "in_dim": 64, "out_dim": 32, "num_latents": 8})
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in lm.parameters():
        p.requires_grad_(False)
    return LMImageVLM(enc, conn, lm, _FakeTok(), num_vision_tokens=8)


def test_forward_loss_generate():
    model = _tiny_model()
    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(3, 60, (2, 6))
    attn = torch.ones_like(input_ids)

    logits = model(images, input_ids, attn)
    assert logits.shape == (2, 8 + 6, 64)            # vision(8) + text(6)

    labels = input_ids.clone(); labels[:, 0] = -100
    loss = model.loss(images, input_ids, attn, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)

    caps = model.generate_caption(images, max_new_tokens=4)
    assert len(caps) == 2 and all(isinstance(c, str) for c in caps)


def test_save_trainable_roundtrip(tmp_path):
    model = _tiny_model()
    path = model.save_trainable(tmp_path / "ck.pt", config={"note": "test"}, step=5)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Only trainable tensors (the connector) are saved — not the frozen LM/encoder.
    assert payload["step"] == 5 and payload["config"]["note"] == "test"
    assert all(k.startswith("connector.") for k in payload["trainable_state"])
    # They load cleanly into a fresh model.
    fresh = _tiny_model()
    missing, unexpected = fresh.load_state_dict(payload["trainable_state"], strict=False)
    assert not unexpected
