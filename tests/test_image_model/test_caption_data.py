"""Caption dataset loader: shapes + label masking, with a fake tokenizer (offline)."""
import torch

from data_pipeline.shard import ShardWriterConfig, write_shards
from data_pipeline.sources import iter_source
from image_model.caption_data import iter_caption_batches, load_caption_dataset


class _FakeTok:
    """Minimal HF-tokenizer-like object for offline tests."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        ids = [(ord(c) % 40) + 3 for c in text.replace(" ", "")][:12]
        return self._Out(ids)


def test_caption_dataset_shapes_and_masking(tmp_path):
    recs = list(iter_source("synthetic", limit=12, seed=0))
    write_shards(recs, tmp_path / "train", ShardWriterConfig(maxcount=100))

    ds = load_caption_dataset(tmp_path, _FakeTok(), image_size=32, split="train", max_len=16)
    n = ds["images"].shape[0]
    assert n == 12
    assert ds["images"].shape[1:] == (3, 32, 32)
    # Stored compactly as uint8 in [0,255]; normalization happens per-batch.
    assert ds["images"].dtype == torch.uint8
    batch = next(iter_caption_batches(ds, batch_size=4, shuffle=False))
    assert batch["images"].dtype == torch.float32
    assert float(batch["images"].min()) >= -1.001 and float(batch["images"].max()) <= 1.001
    assert ds["input_ids"].shape == ds["labels"].shape == (n, 16)

    # BOS position is masked as a target; padding is masked too.
    assert int(ds["labels"][0, 0]) == -100
    assert (ds["labels"] == -100).any()
    # Where attention_mask is 1 (beyond BOS), labels match input_ids.
    row = 0
    for j in range(1, int(ds["attention_mask"][row].sum())):
        assert int(ds["labels"][row, j]) == int(ds["input_ids"][row, j])
