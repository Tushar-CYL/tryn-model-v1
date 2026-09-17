"""Pipeline orchestrator (Week 3, Days 1 & 5).

stream -> clean -> split -> shard, writing training-ready shards plus a run
report and a data card. Runs on the offline synthetic source by default, or a
registered Hub dataset with ``--dataset coco``.

    python -m data_pipeline.build --dataset synthetic --limit 64 --out data/processed/smoke
    python -m data_pipeline.build --dataset coco --limit 500 --out data/processed/coco_v0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from common.logging_utils import get_logger, setup_logging

from .clean import CleanConfig, clean
from .shard import ShardWriterConfig, write_shards
from .sources import DATASETS, iter_source
from .splits import SplitConfig, assign_split, freeze_split

log = get_logger(__name__)


def build(
    dataset: str,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    clean_cfg: CleanConfig | None = None,
    split_cfg: SplitConfig | None = None,
    shard_cfg: ShardWriterConfig | None = None,
    inject_bad: bool = False,
    seed: int = 0,
) -> dict:
    """Run the full pipeline; return a report dict (also written to disk)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_cfg = clean_cfg or CleanConfig()
    split_cfg = split_cfg or SplitConfig()
    shard_cfg = shard_cfg or ShardWriterConfig()

    log.info("streaming '%s' (limit=%s) -> cleaning -> splitting -> sharding", dataset, limit)
    raw = iter_source(dataset, limit=limit, seed=seed, inject_bad=inject_bad)
    clip_scorer = None
    if clean_cfg.clip_min_score is not None:
        from .clip_filter import build_clip_scorer
        log.info("CLIP-score filter on (min=%.3f)", clean_cfg.clip_min_score)
        clip_scorer = build_clip_scorer()
    kept_iter, stats = clean(raw, clean_cfg, clip_scorer=clip_scorer)

    # Route each surviving record to its split, sharding the three streams.
    buckets: dict[str, list] = {"train": [], "val": [], "golden": []}
    kept_ids: list[str] = []
    for rec in kept_iter:
        kept_ids.append(rec.id)
        buckets[assign_split(rec.id, split_cfg)].append(rec)

    shard_manifests = {}
    for split, recs in buckets.items():
        shard_manifests[split] = write_shards(recs, out / split, shard_cfg)

    eval_manifest = freeze_split(kept_ids, out / "eval_split.json", split_cfg)

    report = {
        "dataset": dataset,
        "hub_id": DATASETS[dataset].hub_id if dataset in DATASETS else None,
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
        "clean_stats": stats.as_dict(),
        "split_counts": eval_manifest["counts"],
        "eval_checksum": eval_manifest["checksum"],
        "shards": {k: v["num_shards"] for k, v in shard_manifests.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_data_card(out, report)
    log.info(
        "done: seen=%d kept=%d dropped=%s | splits=%s",
        stats.seen, stats.kept, dict(stats.dropped), eval_manifest["counts"],
    )
    return report


def _write_data_card(out: Path, report: dict) -> None:
    d = report
    lines = [
        f"# Data card — {d['dataset']}",
        "",
        f"- **Built:** {d['built_at']}",
        f"- **Source:** `{d['hub_id'] or 'synthetic (offline)'}`",
        f"- **Records seen:** {d['clean_stats']['seen']} -- **kept:** {d['clean_stats']['kept']}",
        f"- **Dropped:** {d['clean_stats']['dropped'] or 'none'}",
        f"- **Splits:** {d['split_counts']}",
        f"- **Eval checksum:** `{d['eval_checksum'][:16]}...`",
        "",
        "## Layout",
        "```",
        "train/  val/  golden/    # WebDataset .tar shards (jpg + txt + json)",
        "eval_split.json          # frozen val+golden ids + sha256 (never trained on)",
        "report.json              # this run's stats",
        "```",
        "",
        "## Provenance & licensing",
        "Captions/images inherit the upstream dataset's license; verify before",
        "redistribution. The eval split is frozen and checksummed; training code",
        "loads it via `data_pipeline.splits.FrozenEvalSet` and refuses eval ids.",
    ]
    (out / "DATA_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build image training shards.")
    p.add_argument("--dataset", default="synthetic",
                   help=f"'synthetic' or one of {sorted(DATASETS)}")
    p.add_argument("--out", default="data/processed/smoke")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--maxcount", type=int, default=256, help="samples per shard")
    p.add_argument("--inject-bad", action="store_true",
                   help="synthetic only: add dirty records to exercise cleaning")
    p.add_argument("--clip-min-score", type=float, default=None,
                   help="drop image/caption pairs below this CLIP cosine sim (downloads CLIP)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    setup_logging("INFO")
    build(
        args.dataset,
        args.out,
        limit=args.limit,
        clean_cfg=CleanConfig(clip_min_score=args.clip_min_score),
        shard_cfg=ShardWriterConfig(maxcount=args.maxcount),
        inject_bad=args.inject_bad,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
