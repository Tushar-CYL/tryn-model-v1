"""CPU latency / size benchmark for the offline build (Week 7, Day 4).

Reports model size, per-image caption latency, and generation throughput
(tokens/s) on CPU — the numbers that matter for an on-device model. Uses only
stdlib timing + torch so it runs anywhere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import torch

from eval.generate import generate
from image_model.model import ImageVLM

from .export import state_dict_nbytes


@dataclass
class BenchResult:
    model_mb: float
    n_params: int
    latency_ms_per_image: float
    tokens_per_s: float
    batch: int
    max_new_tokens: int

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@torch.no_grad()
def benchmark(
    model: ImageVLM,
    image_size: int = 32,
    batch: int = 8,
    max_new_tokens: int = 20,
    warmup: int = 1,
    iters: int = 3,
) -> BenchResult:
    """Time greedy caption generation on CPU."""
    model = model.eval().cpu()
    images = torch.randn(batch, 3, image_size, image_size)

    for _ in range(warmup):
        generate(model, images, max_new_tokens=max_new_tokens)

    best = float("inf")
    total_tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        out = generate(model, images, max_new_tokens=max_new_tokens)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total_tokens = int((out != 0).sum())  # non-pad tokens produced

    n_params = sum(p.numel() for p in model.parameters())
    return BenchResult(
        model_mb=state_dict_nbytes(model) / (1024 * 1024),
        n_params=n_params,
        latency_ms_per_image=1000.0 * best / batch,
        tokens_per_s=total_tokens / best if best > 0 else 0.0,
        batch=batch,
        max_new_tokens=max_new_tokens,
    )
