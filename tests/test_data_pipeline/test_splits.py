"""Eval split is deterministic, disjoint, frozen, checksummed, and leak-proof."""
from data_pipeline.splits import FrozenEvalSet, SplitConfig, assign_split, freeze_split


def _ids(n):
    return [f"rec-{i:06d}" for i in range(n)]


def test_assignment_is_deterministic_and_stable():
    ids = _ids(500)
    first = {i: assign_split(i) for i in ids}
    # Re-running gives identical assignments; adding data doesn't reshuffle.
    again = {i: assign_split(i) for i in ids + _ids(50)}
    assert all(first[i] == again[i] for i in ids)


def test_freeze_split_disjoint_and_checksummed(tmp_path):
    ids = _ids(1000)
    manifest = freeze_split(ids, tmp_path / "eval_split.json", SplitConfig())

    val = set(manifest["val_ids"])
    golden = set(manifest["golden_ids"])
    assert val.isdisjoint(golden)
    c = manifest["counts"]
    assert c["train"] + c["val"] + c["golden"] == 1000
    assert c["val"] > 0 and c["golden"] > 0  # fractions produced real splits

    frozen = FrozenEvalSet.load(tmp_path / "eval_split.json")
    assert frozen.verify_checksum()


def test_no_leakage_guard(tmp_path):
    ids = _ids(1000)
    freeze_split(ids, tmp_path / "eval_split.json")
    frozen = FrozenEvalSet.load(tmp_path / "eval_split.json")

    eval_ids = [i for i in ids if frozen.is_eval(i)]
    train_ids = [i for i in ids if not frozen.is_eval(i)]

    assert eval_ids and train_ids
    # Training ids pass the guard; an eval id trips it.
    for i in train_ids:
        frozen.assert_not_training(i)
    try:
        frozen.assert_not_training(eval_ids[0])
        raised = False
    except ValueError:
        raised = True
    assert raised
