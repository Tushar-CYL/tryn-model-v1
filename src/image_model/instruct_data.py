"""Instruction / VQA-style data (Phase 2, Week 6, Day 1).

Formats (image, question, answer) turns as chat with answer-only supervision
(via `common.prompt`). The offline synthetic task asks the color or shape of a
per-class scene, so "VQA accuracy" is a genuine, measurable signal on CPU. The
real mix (LLaVA-Instruct + VQAv2 + DocVQA + ChartQA) streams through the same
`data_pipeline` sources and this same chat formatting.
"""
from __future__ import annotations

import torch

from common.prompt import build_chat_example, collate_chat
from common.tokenizer import TinyTokenizer

# class_id -> (color, shape); both words live in the tiny tokenizer charset.
_SCENES = [
    ("red", "circle"),
    ("blue", "square"),
    ("green", "star"),
    ("brown", "house"),
]
_QUESTIONS = {
    "what color is it": 0,   # answer index into the scene tuple
    "what shape is it": 1,
}


def answer_for(class_id: int, question: str) -> str:
    return _SCENES[class_id % len(_SCENES)][_QUESTIONS[question]]


def make_vqa_dataset(
    n_samples: int,
    image_size: int = 32,
    in_channels: int = 3,
    seed: int = 0,
    noise: float = 0.3,
    base_seed: int = 0,
) -> dict:
    """Return {images, input_ids, labels, classes, questions, answers, tokenizer}.

    input_ids/labels are right-padded (labels use -100 for prompt + padding), so
    the batch plugs straight into `ImageVLM.loss(images, input_ids, text_labels=labels)`.

    The per-class base image patterns are keyed by `base_seed`, while `seed`
    varies which samples/noise are drawn. Keep `base_seed` fixed across train and
    held-out so both share the class->image mapping (a real generalisation test);
    change `seed` for fresh, unseen samples of those same classes.
    """
    n_classes = len(_SCENES)
    g = torch.Generator().manual_seed(seed)
    tok = TinyTokenizer()
    q_list = list(_QUESTIONS)

    bg = torch.Generator().manual_seed(base_seed)
    bases = torch.randn(n_classes, in_channels, image_size, image_size, generator=bg)
    classes = torch.randint(0, n_classes, (n_samples,), generator=g)
    q_idx = torch.randint(0, len(q_list), (n_samples,), generator=g)
    images = bases[classes] + noise * torch.randn(
        n_samples, in_channels, image_size, image_size, generator=g
    )

    questions, answers, examples = [], [], []
    for i in range(n_samples):
        q = q_list[int(q_idx[i])]
        a = answer_for(int(classes[i]), q)
        questions.append(q)
        answers.append(a)
        examples.append(build_chat_example(tok, q, a))

    batch = collate_chat(examples)
    return {
        "images": images,
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "classes": classes,
        "questions": questions,
        "answers": answers,
        "tokenizer": tok,
    }


def iter_vqa_batches(dataset: dict, batch_size: int, shuffle: bool = True, seed: int = 0):
    n = dataset["images"].size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle \
        else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start: start + batch_size]
        yield {
            "images": dataset["images"][idx],
            "input_ids": dataset["input_ids"][idx],
            "labels": dataset["labels"][idx],
            "answers": [dataset["answers"][int(j)] for j in idx],
            "questions": [dataset["questions"][int(j)] for j in idx],
        }
