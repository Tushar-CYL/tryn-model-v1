"""BLEU-4: identical text scores high, unrelated text scores low."""
from eval.caption_metrics import corpus_bleu4


def test_identical_scores_high():
    refs = ["a dog runs across a green field", "a red car on the street"]
    assert corpus_bleu4(refs, refs) > 50.0


def test_unrelated_scores_much_lower_than_identical():
    refs = ["a dog runs across a green field today"]
    hyps = ["completely different words here now entirely"]
    unrelated = corpus_bleu4(hyps, refs)
    identical = corpus_bleu4(refs, refs)
    assert unrelated < 20.0           # smoothing inflates short-sentence BLEU
    assert identical > unrelated + 40  # but identical is far higher


def test_empty_is_zero():
    assert corpus_bleu4([], []) == 0.0
