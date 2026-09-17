"""Int8 quantization shrinks the model and keeps accuracy within tolerance."""
import pytest
import torch

from common.seed import set_seed
from eval.evaluate import evaluate
from image_model.data import make_dataset
from image_model.model import ImageVLM
from serving.offline.export import (
    export_quantized,
    onnx_available,
    quantize_dynamic_int8,
    state_dict_nbytes,
)


def _trained_model():
    set_seed(0)
    ds = make_dataset(n_samples=8, n_classes=4, image_size=32, seed=0)
    model = ImageVLM.from_config({
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2},
        "connector": {"type": "resampler", "num_latents": 8, "depth": 1},
        "decoder": {"vocab_size": ds["tokenizer"].vocab_size, "d_model": 64, "depth": 2,
                    "max_seq_len": 128},
    })
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        model.loss(ds["images"], ds["text_ids"]).backward()
        opt.step()
    return model, ds


def test_quantized_is_smaller_and_runs(tmp_path):
    model, ds = _trained_model()
    before = evaluate(model, ds, batch_size=8)

    info = export_quantized(model, tmp_path / "model_int8.pt")
    assert info["int8_bytes"] < info["fp32_bytes"]
    assert info["ratio"] > 1.0
    assert (tmp_path / "model_int8.pt").exists()

    qmodel = quantize_dynamic_int8(model)
    after = evaluate(qmodel, ds, batch_size=8)
    # int8 dynamic quant should barely move accuracy on this tiny model.
    assert before.token_acc - after.token_acc < 0.15


def test_onnx_export_optional(tmp_path):
    if not onnx_available():
        pytest.skip("onnx / onnxruntime not installed")
    import onnxruntime as ort

    from serving.offline.export import export_onnx
    model, _ = _trained_model()
    path = export_onnx(model, tmp_path / "model.onnx", image_size=32, text_len=8)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    import numpy as np
    img = np.random.randn(1, 3, 32, 32).astype("float32")
    txt = np.random.randint(4, 20, (1, 8)).astype("int64")
    (logits,) = sess.run(None, {"images": img, "text_ids": txt})
    assert logits.shape[0] == 1
