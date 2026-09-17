"""Chat template + label masking for image instruction tuning (Week 4, Day 3).

Assembles a (question, answer) turn into decoder token ids and a matching label
tensor where the **prompt is masked (-100) and only the answer is scored**. That
masking is the whole point of a chat template for training: the model learns to
produce answers, not to parrot the question.

The tiny char tokenizer has no dedicated role tokens, so we render roles as
plain text (`question ... answer ...`). A real BPE tokenizer would use special
role/turn tokens instead; the assembly and masking logic are identical.
"""
from __future__ import annotations

import torch

from .tokenizer import BOS, EOS, PAD, TinyTokenizer

# Rendered with charset-safe words (a-z + space) so the tiny tokenizer keeps them
# as real tokens rather than <unk>.
_Q_PREFIX = "question "
_A_PREFIX = " answer "


def build_prompt_ids(tok: TinyTokenizer, question: str) -> list[int]:
    """The prompt prefix `[BOS] question <q> answer ` (no answer tokens yet).

    Generation continues from here to produce the answer.
    """
    return [BOS] + tok.encode(_Q_PREFIX + question + _A_PREFIX,
                              add_bos=False, add_eos=False)


def build_chat_example(
    tok: TinyTokenizer,
    question: str,
    answer: str,
) -> dict:
    """Return {input_ids, labels} (1-D long tensors) for one turn.

    Sequence: [BOS] question <q> answer <a> [EOS]
    Labels:   [-100 over the prompt] <a-ids> [EOS]   (answer-only supervision)
    """
    prompt_ids = build_prompt_ids(tok, question)
    answer_ids = tok.encode(answer, add_bos=False, add_eos=False) + [EOS]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_chat(examples: list[dict]) -> dict:
    """Right-pad a list of build_chat_example() outputs into batched tensors.

    Padded positions are PAD in input_ids and -100 in labels (ignored by loss).
    """
    max_len = max(e["input_ids"].numel() for e in examples)
    b = len(examples)
    input_ids = torch.full((b, max_len), PAD, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        n = e["input_ids"].numel()
        input_ids[i, :n] = e["input_ids"]
        labels[i, :n] = e["labels"]
    return {"input_ids": input_ids, "labels": labels}
