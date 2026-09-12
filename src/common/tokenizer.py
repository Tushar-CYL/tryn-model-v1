"""A tiny, dependency-free char-level tokenizer.

Deliberately minimal — no downloads, deterministic vocab — so the from-scratch
image model and its tests run instantly on CPU. Swap for a real BPE tokenizer
when moving to open decoders in Phase 2.
"""
from __future__ import annotations

import string

PAD, BOS, EOS, UNK = 0, 1, 2, 3
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# A small, fixed printable charset keeps the vocab compact and reproducible.
_CHARS = list(string.ascii_lowercase + string.digits + " .,")


class TinyTokenizer:
    def __init__(self) -> None:
        self.id_to_tok: list[str] = list(_SPECIALS) + _CHARS
        self.tok_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_tok)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_tok)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.tok_to_id.get(ch, UNK) for ch in text.lower()]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            if skip_special and i in (PAD, BOS, EOS, UNK):
                continue
            if 0 <= i < len(self.id_to_tok):
                out.append(self.id_to_tok[i])
        return "".join(out)
