#!/usr/bin/env python3
"""Collect ordinary DiskANN navigation traces for a generic xbin workload.

Supports float32 fbin and uint8 u8bin query files. The query prefix is training
only; deployment/evaluation remains separate.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

THREADS = 4
BEAM = 8


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def itemsize(data_type: str) -> int:
    return {"float32": 4, "uint8": 1}[data_type]


def xbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated xbin")
    return struct.unpack("<II", raw)


def prefix_xbin(src: Path, dst: Path, nrows: int, data_type: str):
    rows, dim = xbin_shape(src)
    if not 0 < nrows <= rows:
        raise ValueError("invalid prefix size")
    size = itemsize(data_type)
    expected = 8 + rows * dim * size
    if src.stat().st_size != expected:
        raise ValueError("query file size mismatch")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * size
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated query file")
            fout.write(b)
            remaining -= len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--data-type", choices=("float32", "uint8"), required=True)
    ap.add_argument("--distance", choices=("inner_product", "squared_l2"), required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--teacher-l", type=int, default=4)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for n in ("binary", "queries", "gt", "index_prefix", "work", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = xbin_shape(args.queries)
    if args.train_rows > rows:
        raise ValueError("training prefix exceeds queries")
    ext = ".fbin" if args.data_type == "float32" else ".u8bin"
    train = args.work / ("train" + ext)
    prefix_xbin(args.queries, train, args.train_rows, args.data_type)

    trace = args.out / "teacher.jsonl"
    output = args.out / "teacher-output.json"
    inp = args.out / "teacher-input.json"
    cfg = {
        "search_directories": [str(args.work)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": args.data_type,
                    "load_path": str(args.index_prefix),
                },
                "search_phase": {
                    "queries": str(train),
                    "groundtruth": str(args.gt),
                    "search_list": [args.teacher_l],
                    "beam_width": BEAM,
                    "recall_at": 1,
                    "num_threads": THREADS,
                    "is_flat_search": False,
                    "distance": args.distance,
                    "vector_filters_file": None,
                    "num_nodes_to_cache": None,
                    "search_io_limit": None,
                    "post_processor": None,
                },
            },
        }],
    }
    save(inp, cfg)

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    env["DISKANN_TRACE_FILE"] = str(trace)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_QSEV_FILE",
    ):
        env.pop(name, None)

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (args.out / "teacher.log").open("w") as log:
            subprocess.run(
                [str(args.binary), "run", "--input-file", str(inp), "--output-file", str(output)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
            )

    records = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    if len(records) != args.train_rows:
        raise ValueError(f"expected {args.train_rows} traces, got {len(records)}")
    if [int(r["query"]) for r in records] != list(range(args.train_rows)):
        raise ValueError("trace query numbering mismatch")
    lens = np.asarray([len(r["ids"]) for r in records], dtype=np.int64)
    ios = np.asarray([int(r["io_count"]) for r in records], dtype=np.int64)
    result = {
        "queries": args.train_rows,
        "dimension": dim,
        "data_type": args.data_type,
        "distance": args.distance,
        "teacher_l": args.teacher_l,
        "trace_file": trace.name,
        "mean_expanded_vertices": float(lens.mean()),
        "median_expanded_vertices": float(np.median(lens)),
        "p95_expanded_vertices": float(np.quantile(lens, 0.95)),
        "mean_io": float(ios.mean()),
        "median_io": float(np.median(ios)),
        "p95_io": float(np.quantile(ios, 0.95)),
    }
    save(args.out / "trace-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
