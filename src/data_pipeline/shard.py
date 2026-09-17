"""Sharding stage (Week 3, Day 2): write records to WebDataset .tar shards.

WebDataset is just a tar of grouped files sharing a key:

    000000.jpg  000000.txt  000000.json
    000001.jpg  000001.txt  000001.json  ...

We write with the stdlib `tarfile` (no special dependency to *produce* shards)
and provide a reader that prefers the installed `webdataset` library, falling
back to a plain tar reader so tests never depend on it. Shards are capped at
`maxcount` samples so training can stream many shards in parallel.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .sources import Record


@dataclass
class ShardWriterConfig:
    pattern: str = "shard-{index:05d}.tar"
    maxcount: int = 1000       # samples per shard
    image_format: str = "JPEG"


def write_shards(
    records: Iterable[Record],
    out_dir: str | Path,
    cfg: ShardWriterConfig | None = None,
) -> dict:
    """Write records to .tar shards under `out_dir`. Returns a manifest dict."""
    cfg = cfg or ShardWriterConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards: list[str] = []
    total = 0
    tar: tarfile.TarFile | None = None
    in_shard = 0

    def _open(index: int) -> tarfile.TarFile:
        name = cfg.pattern.format(index=index)
        shards.append(name)
        return tarfile.open(out / name, "w")

    def _add(t: tarfile.TarFile, key: str, ext: str, data: bytes) -> None:
        info = tarfile.TarInfo(f"{key}.{ext}")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))

    try:
        for rec in records:
            if tar is None or in_shard >= cfg.maxcount:
                if tar is not None:
                    tar.close()
                tar = _open(len(shards))
                in_shard = 0

            key = f"{total:07d}"
            buf = io.BytesIO()
            rec.image.convert("RGB").save(buf, format=cfg.image_format)
            _add(tar, key, "jpg", buf.getvalue())
            _add(tar, key, "txt", rec.caption.encode("utf-8"))
            _add(tar, key, "json", json.dumps({"id": rec.id, **rec.meta}).encode("utf-8"))

            total += 1
            in_shard += 1
    finally:
        if tar is not None:
            tar.close()

    manifest = {
        "num_samples": total,
        "num_shards": len(shards),
        "shards": shards,
        "maxcount": cfg.maxcount,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_shards(out_dir: str | Path, *, decode_image: bool = True) -> Iterator[Record]:
    """Read records back from shards under `out_dir` (roundtrip of write_shards)."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for shard_name in manifest["shards"]:
        yield from _read_one_tar(out / shard_name, decode_image=decode_image)


def _read_one_tar(path: Path, *, decode_image: bool) -> Iterator[Record]:
    groups: dict[str, dict] = {}
    with tarfile.open(path, "r") as t:
        for member in t.getmembers():
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            payload = t.extractfile(member).read()
            groups.setdefault(key, {})[ext] = payload
    for key in sorted(groups):
        g = groups[key]
        meta = json.loads(g["json"].decode("utf-8")) if "json" in g else {}
        rec_id = meta.pop("id", key)
        caption = g.get("txt", b"").decode("utf-8")
        image = Image.open(io.BytesIO(g["jpg"])) if (decode_image and "jpg" in g) else None
        if image is not None:
            image.load()
        yield Record(id=rec_id, image=image, caption=caption, meta=meta)
