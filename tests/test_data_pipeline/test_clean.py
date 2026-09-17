"""Cleaning drops the right records and reports accurate stats."""
from data_pipeline.clean import CleanConfig, clean
from data_pipeline.sources import iter_source


def test_clean_drops_dirty_records():
    # 8 clean synthetic records + 4 injected dirty ones (dup, tiny, non-en, empty).
    raw = list(iter_source("synthetic", limit=8, seed=0, inject_bad=True))
    kept_iter, stats = clean(raw, CleanConfig())
    kept = list(kept_iter)  # consume so stats populate

    assert stats.seen == len(raw)
    assert stats.kept == len(kept)
    # Each dirty category should have been caught at least once.
    assert stats.dropped["dup_caption"] >= 1
    assert stats.dropped["image_too_small"] >= 1
    assert stats.dropped["non_english"] >= 1
    assert stats.dropped["caption_too_short"] >= 1  # the "  " empty caption
    # Survivors are all valid.
    assert all(len(r.caption) >= CleanConfig().min_caption_chars for r in kept)
    assert all(min(r.image.size) >= CleanConfig().min_size for r in kept)


def test_clean_dedup_captions():
    # First 4 synthetic captions are unique and cycle; limit=4 -> all unique.
    raw = list(iter_source("synthetic", limit=4, seed=0))
    kept, stats = clean(raw, CleanConfig())
    assert len(list(kept)) == 4
    assert stats.dropped.get("dup_caption", 0) == 0
