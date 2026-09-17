"""VQA accuracy + majority baseline are well-formed and consistent."""
from common.seed import set_seed
from eval.baseline import compare_to_baseline, majority_answer_accuracy
from eval.vqa import vqa_accuracy
from image_model.instruct_data import make_vqa_dataset
from image_model.model import ImageVLM


def _model(vocab):
    return ImageVLM.from_config({
        "encoder": {"type": "tiny", "image_size": 32, "patch_size": 8, "d_model": 64, "depth": 2},
        "connector": {"type": "resampler", "num_latents": 8, "depth": 1},
        "decoder": {"vocab_size": vocab, "d_model": 64, "depth": 2, "max_seq_len": 128},
    })


def test_vqa_accuracy_range():
    set_seed(0)
    ds = make_vqa_dataset(24, seed=0)
    rep = vqa_accuracy(_model(ds["tokenizer"].vocab_size), ds)
    assert 0.0 <= rep.accuracy <= 1.0
    assert rep.n == 24 and rep.samples
    assert set(rep.samples[0]) == {"q", "gold", "pred"}


def test_majority_baseline_is_frequency_of_top_answer():
    ds = make_vqa_dataset(40, seed=1)
    base = majority_answer_accuracy(ds)
    # majority accuracy == share of the single most common gold answer
    from collections import Counter
    golds = [a.strip().lower() for a in ds["answers"]]
    top = Counter(golds).most_common(1)[0][1]
    assert abs(base.accuracy - top / len(golds)) < 1e-9


def test_compare_structure():
    ds = make_vqa_dataset(24, seed=0)
    cmp = compare_to_baseline(_model(ds["tokenizer"].vocab_size), ds)
    assert set(cmp) >= {"model_accuracy", "baseline_accuracy", "beats_baseline", "baseline"}
    assert isinstance(cmp["beats_baseline"], bool)
