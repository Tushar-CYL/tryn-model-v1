"""Chat template masks the prompt and supervises only the answer."""
import torch

from common.prompt import build_chat_example, collate_chat
from common.tokenizer import EOS, TinyTokenizer


def test_prompt_is_masked_answer_is_scored():
    tok = TinyTokenizer()
    ex = build_chat_example(tok, "what color is it", "red")

    ids, labels = ex["input_ids"], ex["labels"]
    assert ids.shape == labels.shape

    # Prompt positions are ignored (-100); the tail (answer + EOS) is scored.
    scored = labels != -100
    assert scored.sum() > 0
    # Every scored label matches the input id at that position (teacher forcing).
    assert torch.equal(labels[scored], ids[scored])
    # The last scored token is EOS.
    assert int(ids[scored][-1]) == EOS
    # The leading positions (the question) are masked.
    assert int(labels[0]) == -100


def test_collate_pads_and_masks():
    tok = TinyTokenizer()
    batch = collate_chat([
        build_chat_example(tok, "what color is it", "red"),
        build_chat_example(tok, "how many", "two dogs on a mat"),
    ])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].size(0) == 2
    # Padding on the shorter row shows up as -100 in labels.
    assert (batch["labels"] == -100).any()
