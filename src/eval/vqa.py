"""VQA accuracy for the instruction-tuned image model (Week 6, Days 3-4).

For each (image, question) the model greedily generates an answer; we normalise
(lowercase, strip) and compare to the gold answer. This is the held-out metric
the instruction-tuning gate is judged on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common.prompt import build_prompt_ids
from common.tokenizer import TinyTokenizer
from image_model.model import ImageVLM

from .generate import generate_from_prompt


def _normalise(text: str) -> str:
    return text.strip().lower()


@dataclass
class VQAReport:
    accuracy: float
    n: int
    samples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"vqa_accuracy": round(self.accuracy, 4), "n": self.n}


@torch.no_grad()
def vqa_accuracy(
    model: ImageVLM,
    dataset: dict,
    max_new_tokens: int = 12,
    n_samples: int = 4,
    device=None,
) -> VQAReport:
    """Evaluate a VQA dataset ({images, questions, answers, tokenizer}).

    Pass `device` explicitly (from the trainer) to guarantee inputs land on the
    model's device; falls back to inferring it from the model's parameters.
    """
    model.eval()
    tok: TinyTokenizer = dataset["tokenizer"]
    if device is None:
        device = next(model.parameters()).device
    images = dataset["images"].to(device)
    questions = dataset["questions"]
    answers = dataset["answers"]

    hits = 0
    samples: list[dict] = []
    for i in range(len(questions)):
        prompt = build_prompt_ids(tok, questions[i])
        gen_ids = generate_from_prompt(model, images[i], prompt, max_new_tokens)
        pred = _normalise(tok.decode(gen_ids))
        gold = _normalise(answers[i])
        if pred == gold:
            hits += 1
        if len(samples) < n_samples:
            samples.append({"q": questions[i], "gold": gold, "pred": pred})

    n = len(questions)
    return VQAReport(accuracy=hits / max(n, 1), n=n, samples=samples)
