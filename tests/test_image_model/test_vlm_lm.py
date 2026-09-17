"""LM-backed VLM smoke (opt-in: downloads SmolLM2). Set RUN_LM_TESTS=1 to run.

Skipped by default so CI stays fast/offline; the path is exercised on Kaggle and
in the local smoke. When enabled, it checks shapes, a loss step, and generation.
"""
import os

import pytest
import torch

RUN = os.environ.get("RUN_LM_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="set RUN_LM_TESTS=1 (downloads SmolLM2)")


def _cfg():
    return {
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2},
        "connector": {"type": "resampler", "num_latents": 8, "depth": 1},
        "lm": {"model_name": "HuggingFaceTB/SmolLM2-135M", "freeze_base": True},
    }


def test_lm_vlm_forward_and_generate():
    from image_model.vlm_lm import LMImageVLM

    model = LMImageVLM.from_pretrained(_cfg())
    tok = model.tokenizer
    images = torch.randn(2, 3, 32, 32)
    enc = tok(["a red circle", "a blue square"], return_tensors="pt", padding=True)

    logits = model(images, enc.input_ids, enc.attention_mask)
    assert logits.shape[0] == 2
    assert logits.shape[1] == model.num_vision_tokens + enc.input_ids.shape[1]

    labels = enc.input_ids.clone(); labels[enc.attention_mask == 0] = -100
    loss = model.loss(images, enc.input_ids, enc.attention_mask, labels)
    assert torch.isfinite(loss)

    caps = model.generate_caption(images, max_new_tokens=5)
    assert len(caps) == 2 and all(isinstance(c, str) for c in caps)
