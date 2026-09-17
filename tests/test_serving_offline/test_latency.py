"""CPU benchmark returns sane, well-formed numbers."""
from common.tokenizer import TinyTokenizer
from image_model.model import ImageVLM
from serving.offline.bench import benchmark


def _model():
    return ImageVLM.from_config({
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2},
        "connector": {"type": "resampler", "num_latents": 8, "depth": 1},
        "decoder": {"vocab_size": TinyTokenizer().vocab_size, "d_model": 64, "depth": 2,
                    "max_seq_len": 128},
    })


def test_benchmark_shape():
    res = benchmark(_model(), batch=4, max_new_tokens=8, iters=2)
    d = res.as_dict()
    assert d["n_params"] > 0
    assert d["model_mb"] > 0
    assert d["latency_ms_per_image"] > 0
    assert d["tokens_per_s"] > 0
    assert d["batch"] == 4 and d["max_new_tokens"] == 8
