"""Connector maps encoder features -> decoder token space with correct shapes."""
import torch

from common.connector import MLPConnector, PerceiverResampler, build_connector


def test_connector_shapes():
    b, n, in_dim, out_dim = 4, 16, 48, 64
    conn = MLPConnector(in_dim=in_dim, out_dim=out_dim, hidden_dim=96)
    x = torch.randn(b, n, in_dim)
    y = conn(x)
    assert y.shape == (b, n, out_dim)


def test_connector_is_trainable():
    conn = MLPConnector(8, 8)
    assert all(p.requires_grad for p in conn.parameters())


def test_resampler_fixes_token_count():
    # 196 encoder tokens (SigLIP-sized) -> a fixed 16-token budget.
    b, n, in_dim, out_dim, latents = 3, 196, 768, 64, 16
    res = PerceiverResampler(in_dim, out_dim, num_latents=latents)
    y = res(torch.randn(b, n, in_dim))
    assert y.shape == (b, latents, out_dim)
    assert res.out_tokens == latents
    # Output count is independent of input length.
    y2 = res(torch.randn(b, 49, in_dim))
    assert y2.shape == (b, latents, out_dim)


def test_build_connector_factory():
    mlp = build_connector({"type": "mlp", "in_dim": 32, "out_dim": 64})
    assert isinstance(mlp, MLPConnector)
    res = build_connector(
        {"type": "resampler", "in_dim": 32, "out_dim": 64, "num_latents": 8}
    )
    assert isinstance(res, PerceiverResampler) and res.out_tokens == 8
