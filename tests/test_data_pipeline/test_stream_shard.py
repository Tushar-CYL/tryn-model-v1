"""Streaming source -> shard writer -> reader roundtrip (offline, synthetic)."""
from data_pipeline.shard import ShardWriterConfig, read_shards, write_shards
from data_pipeline.sources import iter_source


def test_synthetic_source_shape():
    recs = list(iter_source("synthetic", limit=10, seed=0))
    assert len(recs) == 10
    r = recs[0]
    assert r.image.size == (32, 32)
    assert isinstance(r.caption, str) and r.caption
    assert r.id


def test_shard_roundtrip(tmp_path):
    recs = list(iter_source("synthetic", limit=20, seed=1))
    manifest = write_shards(recs, tmp_path, ShardWriterConfig(maxcount=7))

    assert manifest["num_samples"] == 20
    assert manifest["num_shards"] == 3  # ceil(20/7)

    back = list(read_shards(tmp_path))
    assert len(back) == 20
    # captions survive the roundtrip; order is preserved by the 7-digit key
    assert [r.caption for r in back] == [r.caption for r in recs]
    assert back[0].image.size == (32, 32)
