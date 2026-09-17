"""Optimizer grouping + cosine-warmup schedule behave correctly."""
from torch import nn

from common.optim import build_optimizer, cosine_warmup


def test_cosine_warmup_shape():
    f = cosine_warmup(total_steps=100, warmup_steps=10, min_ratio=0.1)
    assert f(0) == 0.0                      # warmup starts at 0
    assert abs(f(10) - 1.0) < 1e-6          # peak at end of warmup
    assert f(5) < f(10)                     # rising during warmup
    assert f(100) < f(10)                   # decayed by the end
    assert abs(f(100) - 0.1) < 1e-6         # floors at min_ratio


def test_build_optimizer_groups():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.05)
    assert len(opt.param_groups) == 2
    decay = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    no_decay = next(g for g in opt.param_groups if g["weight_decay"] == 0)
    # 2-D weights get decay; biases + LayerNorm params do not.
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
