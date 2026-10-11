#!/usr/bin/env python3
"""Download and freeze public BIGANN-benchmark subsets for NavHints evaluation.

Supported datasets:
  text2image-10M  float32, d=200, inner product, 100K public queries
  bigann-10M      uint8,   d=128, squared L2, 10K public queries
  bigann-100M     uint8,   d=128, squared L2, 10K public queries

The first 5K public queries are training history; rows [5000,10000) are the
frozen 5K evaluation set. Ground truth is sliced from the official public GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

DATASETS = {
    "text2image-10M": {
        "rows": 10_000_000,
        "dim": 200,
        "dtype": "float32",
        "itemsize": 4,
        "metric": "inner_product",
        "data_type": "float32",
        "base_url": "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin",
        "query_url": "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
        "gt_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/text2image-10M",
        "base_source_rows": 1_000_000_000,
        "query_rows": 100_000,
        "base_ext": ".fbin",
        "query_ext": ".fbin",
    },
    "bigann-10M": {
        "rows": 10_000_000,
        "dim": 128,
        "dtype": "uint8",
        "itemsize": 1,
        "metric": "squared_l2",
        "data_type": "uint8",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
        "query_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
        "gt_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M",
        "base_source_rows": 1_000_000_000,
        "query_rows": 10_000,
        "base_ext": ".u8bin",
        "query_ext": ".u8bin",
    },
    "bigann-100M": {
        "rows": 100_000_000,
        "dim": 128,
        "dtype": "uint8",
        "itemsize": 1,
        "metric": "squared_l2",
        "data_type": "uint8",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
        "query_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
        "gt_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M",
        "base_source_rows": 1_000_000_000,
        "query_rows": 10_000,
        "base_ext": ".u8bin",
        "query_ext": ".u8bin",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, max_bytes: int | None = None) -> None:
    if dst.is_file() and (max_bytes is None or dst.stat().st_size == max_bytes):
        return
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    print(f"download {url} -> {dst}", flush=True)
    t0 = time.time()
    total = 0
    with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as out:
        while True:
            need = (1 << 20) if max_bytes is None else min(1 << 20, max_bytes - total)
            if need <= 0:
                break
            block = src.read(need)
            if not block:
                break
            out.write(block)
            total += len(block)
            if total and total % (256 << 20) < (1 << 20):
                dt = max(time.time() - t0, 1e-9)
                print(f"  {total / 2**30:.2f} GiB at {total / 2**20 / dt:.1f} MiB/s", flush=True)
    if max_bytes is not None and total != max_bytes:
        raise RuntimeError(f"cropped download got {total} bytes, expected {max_bytes}")
    tmp.replace(dst)


def read_xbin_header(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"bad header: {path}")
    return struct.unpack("<II", raw)


def validate_xbin(path: Path, rows: int, dim: int, itemsize: int) -> None:
    got_rows, got_dim = read_xbin_header(path)
    if (got_rows, got_dim) != (rows, dim):
        raise ValueError(f"{path}: header {(got_rows, got_dim)} != {(rows, dim)}")
    expected = 8 + rows * dim * itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: size {path.stat().st_size} != {expected}")


def crop_base(cfg, dst: Path) -> None:
    wanted = 8 + cfg["rows"] * cfg["dim"] * cfg["itemsize"]
    # The first successful preparation rewrites the downloaded 1B header to
    # the frozen subset size. Treat that already-cropped file as final instead
    # of interpreting it as a malformed 1B source on later evaluation runs.
    if dst.is_file() and dst.stat().st_size == wanted:
        rows, dim = read_xbin_header(dst)
        if (rows, dim) == (cfg["rows"], cfg["dim"]):
            validate_xbin(dst, cfg["rows"], cfg["dim"], cfg["itemsize"])
            return
    download(cfg["base_url"], dst, max_bytes=wanted)
    src_rows, src_dim = read_xbin_header(dst)
    if src_rows != cfg["base_source_rows"] or src_dim != cfg["dim"]:
        raise ValueError(f"unexpected source header {(src_rows, src_dim)}")
    with dst.open("r+b") as f:
        f.write(struct.pack("<II", cfg["rows"], cfg["dim"]))
    validate_xbin(dst, cfg["rows"], cfg["dim"], cfg["itemsize"])


def slice_xbin(src: Path, dst: Path, start: int, count: int, itemsize: int) -> None:
    rows, dim = read_xbin_header(src)
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError("invalid xbin slice")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * itemsize)
        fout.write(struct.pack("<II", count, dim))
        remaining = count * dim * itemsize
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated xbin slice")
            fout.write(b)
            remaining -= len(b)


def validate_gt(path: Path, min_rows: int) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated GT")
    rows, k = struct.unpack("<II", raw)
    expected = 8 + rows * k * 8
    if path.stat().st_size != expected:
        raise ValueError(f"GT size mismatch: {path.stat().st_size} != {expected}")
    if rows < min_rows or k < 10:
        raise ValueError(f"GT shape {(rows, k)} insufficient")
    return rows, k


def slice_gt(src: Path, dst: Path, start: int, count: int) -> None:
    rows, k = validate_gt(src, start + count)
    row_bytes = k * 4
    ids_bytes = rows * row_bytes
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<II", count, k))
        fin.seek(8 + start * row_bytes)
        remaining = count * row_bytes
        while remaining:
            b = fin.read(min(8 << 20, remaining))
            if not b:
                raise ValueError("truncated GT ids")
            fout.write(b)
            remaining -= len(b)
        fin.seek(8 + ids_bytes + start * row_bytes)
        remaining = count * row_bytes
        while remaining:
            b = fin.read(min(8 << 20, remaining))
            if not b:
                raise ValueError("truncated GT distances")
            fout.write(b)
            remaining -= len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = dict(DATASETS[args.dataset])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    base = out / ("base" + cfg["base_ext"])
    queries = out / ("queries" + cfg["query_ext"])
    gt = out / "groundtruth.bin"

    crop_base(cfg, base)
    query_bytes = 8 + cfg["query_rows"] * cfg["dim"] * cfg["itemsize"]
    download(cfg["query_url"], queries, max_bytes=query_bytes)
    validate_xbin(queries, cfg["query_rows"], cfg["dim"], cfg["itemsize"])
    download(cfg["gt_url"], gt)
    gt_rows, gt_k = validate_gt(gt, 10_000)

    train = out / ("train5000" + cfg["query_ext"])
    held = out / ("heldout5000" + cfg["query_ext"])
    held_gt = out / "heldout5000.gt"
    slice_xbin(queries, train, 0, 5000, cfg["itemsize"])
    slice_xbin(queries, held, 5000, 5000, cfg["itemsize"])
    slice_gt(gt, held_gt, 5000, 5000)

    result = {
        "dataset": args.dataset,
        **{k: cfg[k] for k in (
            "rows", "dim", "dtype", "metric", "data_type", "query_rows",
            "base_url", "query_url", "gt_url",
        )},
        "official_gt_rows": gt_rows,
        "official_gt_k": gt_k,
        "split": {"train": [0, 5000], "heldout": [5000, 10000]},
        "files": {
            "base": str(base),
            "queries": str(queries),
            "groundtruth": str(gt),
            "train5000": str(train),
            "heldout5000": str(held),
            "heldout5000_gt": str(held_gt),
        },
        "sha256": {
            "base": sha256(base),
            "queries": sha256(queries),
            "groundtruth": sha256(gt),
            "train5000": sha256(train),
            "heldout5000": sha256(held),
            "heldout5000_gt": sha256(held_gt),
        },
    }
    (out / "dataset.manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
