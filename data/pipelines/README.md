# data/pipelines

Image data pipeline for Phase 2. **Code lives in `src/data_pipeline/`** (importable,
tested under `tests/test_data_pipeline/`); this folder holds the design notes and is
where generated shards/cards land under `data/processed/<name>/`.

## Flow

    stream (sources.py) -> clean (clean.py) -> split (splits.py) -> shard (shard.py)

- **stream** — `iter_source(name, limit)`. `name='synthetic'` runs offline; a registered
  name (`coco`, `coco_karpathy`, `cc3m`) streams from the HuggingFace Hub. Each item is a
  `Record = {id, image (PIL), caption, meta}`.
- **clean** — drop broken/tiny images, empty/too-long captions, non-English captions
  (ASCII-ratio heuristic), and exact caption/image duplicates. Optional CLIP-score hook.
  Returns survivors + a `CleanStats` (kept/dropped counts).
- **split** — hash each id into train/val/golden. Deterministic and stable as data grows;
  `freeze_split` writes a checksummed manifest; `FrozenEvalSet` guards against leakage.
- **shard** — write WebDataset `.tar` shards (`<key>.jpg` + `.txt` + `.json`) with a
  `manifest.json`; `read_shards` roundtrips them back to `Record`s.

## Build shards

```bash
# Offline smoke (no network):
python -m data_pipeline.build --dataset synthetic --limit 64 --out data/processed/smoke --inject-bad

# Real COCO slice (streams from the Hub):
python -m data_pipeline.build --dataset coco --limit 500 --out data/processed/coco_v0
```

Each run writes `train/ val/ golden/` shards, `eval_split.json` (frozen + sha256),
`report.json`, and `DATA_CARD.md`. See MASTER_PLAN.md §8.
