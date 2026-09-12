"""Connector maps encoder features -> decoder token space with correct shapes."""
import torch

from common.connector import MLPConnector


def test_connector_shapes():
    b, n, in_dim, out_dim = 4, 16, 48, 64
    conn = MLPConnector(in_dim=in_dim, out_dim=out_dim, hidden_dim=96)
    x = torch.randn(b, n, in_dim)
    y = conn(x)
    assert y.shape == (b, n, out_dim)


def test_connector_is_trainable():
    conn = MLPConnector(8, 8)
    assert all(p.requires_grad for p in conn.parameters())
