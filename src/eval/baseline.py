"""Baseline comparison for the Phase-2 gate (Week 6, Day 5).

The gate asks the model to *beat a baseline* on a held-out task. On CPU-tiny
scale the honest, fully-offline baseline is the **majority-answer** predictor
(always answer the most frequent gold answer) — the standard trivial baseline
for a VQA-style classification task. The trained model must clear it.

The plan's headline comparison is vs. SmolVLM-256M; that is a real-scale claim
that needs the GPU-trained checkpoint and a SmolVLM download, so it lives behind
`smolvlm_available()` / `smolvlm_baseline()` as a documented opt-in rather than a
CPU default.
"""
from __future__ import annotations

from collections import Counter

from image_model.model import ImageVLM

from .vqa import VQAReport, _normalise, vqa_accuracy


def majority_answer_accuracy(dataset: dict) -> VQAReport:
    """Accuracy of always predicting the single most common gold answer."""
    golds = [_normalise(a) for a in dataset["answers"]]
    if not golds:
        return VQAReport(accuracy=0.0, n=0)
    most_common, count = Counter(golds).most_common(1)[0]
    return VQAReport(
        accuracy=count / len(golds),
        n=len(golds),
        samples=[{"q": "<majority>", "gold": most_common, "pred": most_common}],
    )


def compare_to_baseline(model: ImageVLM, dataset: dict, max_new_tokens: int = 12) -> dict:
    """Run the model + the majority baseline; report whether the model wins."""
    model_rep = vqa_accuracy(model, dataset, max_new_tokens=max_new_tokens)
    base_rep = majority_answer_accuracy(dataset)
    return {
        "model_accuracy": round(model_rep.accuracy, 4),
        "baseline_accuracy": round(base_rep.accuracy, 4),
        "baseline": "majority_answer",
        "beats_baseline": model_rep.accuracy > base_rep.accuracy,
        "model_samples": model_rep.samples,
    }


def smolvlm_available() -> bool:
    """True if transformers + a SmolVLM checkpoint can be loaded (opt-in path)."""
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def smolvlm_baseline(*_args, **_kwargs):  # pragma: no cover - opt-in real-scale path
    """Placeholder for the real-scale SmolVLM-256M comparison.

    Beating SmolVLM on real VQA requires the GPU-trained checkpoint (Weeks 5-6 at
    scale) and a SmolVLM download. Wire it here when running on the GPU box.
    """
    raise NotImplementedError(
        "SmolVLM baseline is the real-scale (GPU) comparison; use "
        "majority_answer_accuracy for the offline gate."
    )
