"""Held-out evaluation for the image model (Phase 2, Week 5, Day 4).

Computes loss + next-token accuracy over a dataset, exact-match of greedy
generations against the reference captions, and returns a few qualitative
samples (image -> generated text vs. target) for eyeballing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common.decoder import lm_loss
from common.tokenizer import TinyTokenizer
from image_model.data import iter_batches
from image_model.model import ImageVLM

from .generate import generate
from .metrics import exact_match, token_accuracy


@dataclass
class EvalReport:
    loss: float
    token_acc: float
    exact_match: float
    n: int
    samples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "loss": round(self.loss, 4),
            "token_acc": round(self.token_acc, 4),
            "exact_match": round(self.exact_match, 4),
            "n": self.n,
        }


@torch.no_grad()
def evaluate(
    model: ImageVLM,
    dataset: dict,
    batch_size: int = 16,
    max_new_tokens: int = 24,
    n_samples: int = 4,
) -> EvalReport:
    """Evaluate `model` on `dataset` ({images, text_ids, tokenizer})."""
    model.eval()
    tok: TinyTokenizer = dataset["tokenizer"]
    device = next(model.parameters()).device

    total_loss, total_tok_correct, total_tok = 0.0, 0.0, 0
    n_batches = 0
    for batch in iter_batches(dataset, batch_size, shuffle=False):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        logits = model.forward(images, text_ids)
        labels = model.build_labels(text_ids)
        total_loss += float(lm_loss(logits, labels))
        n_batches += 1

        # accumulate token accuracy weighted by scored positions
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        correct = (logits[:, :-1, :].argmax(-1) == shift_labels) & mask
        total_tok_correct += float(correct.sum())
        total_tok += int(mask.sum())

    # exact-match via greedy generation over the whole set
    preds = generate(model, dataset["images"].to(device), max_new_tokens=max_new_tokens)
    em = exact_match(preds, dataset["text_ids"])

    samples = []
    for i in range(min(n_samples, dataset["images"].size(0))):
        samples.append({
            "target": tok.decode(dataset["text_ids"][i].tolist()),
            "generated": tok.decode(preds[i].tolist()),
        })

    n = dataset["images"].size(0)
    return EvalReport(
        loss=total_loss / max(n_batches, 1),
        token_acc=(total_tok_correct / total_tok) if total_tok else 0.0,
        exact_match=em,
        n=n,
        samples=samples,
    )
