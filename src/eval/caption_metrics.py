"""Lightweight caption metrics (no external deps): corpus BLEU-4.

Single reference per hypothesis (our shards store one caption per image), with
add-1 smoothing so short corpora don't collapse to 0. Good enough to track
caption quality trend during training; swap in pycocoevalcap CIDEr for a paper
number later.
"""
from __future__ import annotations

from collections import Counter


def _tokens(s: str) -> list[str]:
    return s.lower().split()


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Corpus BLEU-4 (%) with add-1 smoothing and brevity penalty."""
    import math

    if not hypotheses:
        return 0.0
    weights = [0.25, 0.25, 0.25, 0.25]
    p_log_sum = 0.0
    hyp_len_total, ref_len_total = 0, 0

    for n, w in enumerate(weights, start=1):
        clipped, total = 0, 0
        for hyp, ref in zip(hypotheses, references):
            h, r = _tokens(hyp), _tokens(ref)
            hc, rc = _ngram_counts(h, n), _ngram_counts(r, n)
            for g, c in hc.items():
                clipped += min(c, rc.get(g, 0))
            total += max(0, len(h) - n + 1)
        # add-1 smoothing
        precision = (clipped + 1) / (total + 1)
        p_log_sum += w * math.log(precision)

    for hyp, ref in zip(hypotheses, references):
        hyp_len_total += len(_tokens(hyp))
        ref_len_total += len(_tokens(ref))
    if hyp_len_total == 0:
        return 0.0
    bp = 1.0 if hyp_len_total > ref_len_total else math.exp(1 - ref_len_total / hyp_len_total)
    return round(100.0 * bp * math.exp(p_log_sum), 2)
