"""Offline export + quantization (Phase 2, Week 7, Days 1-2).

Primary path (no extra deps): PyTorch **dynamic int8** quantization of the
Linear layers, which is the right fit for a small transformer running on CPU —
weights become int8, activations are quantized on the fly. We measure the
serialized size before/after and let callers check the accuracy drop.

ONNX export is provided as a best-effort second path (`export_onnx`); it is
skipped when `onnx`/`onnxruntime` aren't installed.
"""
from __future__ import annotations

import io
from pathlib import Path

import torch
from torch import nn

from image_model.model import ImageVLM


def state_dict_nbytes(model: nn.Module) -> int:
    """Serialized size of the model's state dict, in bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def quantize_dynamic_int8(model: ImageVLM) -> nn.Module:
    """Dynamic int8 quantization over `nn.Linear` (connector, MLP, LM head).

    nn.MultiheadAttention's `out_proj` is intentionally excluded by PyTorch, so
    attention stays float — expected and fine for a small model.
    """
    model = model.eval().cpu()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def export_quantized(model: ImageVLM, path: str | Path) -> dict:
    """Quantize and save. Returns {fp32_bytes, int8_bytes, ratio, path}."""
    fp32 = state_dict_nbytes(model)
    qmodel = quantize_dynamic_int8(model)
    int8 = state_dict_nbytes(qmodel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), path)
    return {
        "fp32_bytes": fp32,
        "int8_bytes": int8,
        "ratio": round(fp32 / max(int8, 1), 3),
        "path": str(path),
    }


def onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def export_onnx(
    model: ImageVLM,
    path: str | Path,
    image_size: int = 32,
    text_len: int = 8,
) -> str:
    """Best-effort ONNX export of the full forward(images, text_ids).

    Raises RuntimeError if ONNX isn't installed; wrap in `onnx_available()`.
    """
    if not onnx_available():
        raise RuntimeError("onnx / onnxruntime not installed.")
    model = model.eval().cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy_img = torch.randn(1, 3, image_size, image_size)
    dummy_txt = torch.randint(4, 20, (1, text_len))
    torch.onnx.export(
        model,
        (dummy_img, dummy_txt),
        str(path),
        input_names=["images", "text_ids"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "text_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript exporter (no onnxscript dependency)
    )
    return str(path)
