#!/usr/bin/env python3
"""Collect medoid-start DiskANN traversal traces for MemGraph training.

This uses the existing traversal-recorder patch with no portal, Catapult, node
cache, or custom starts enabled. The first 5000 workload queries are training
only; heldout queries are untouched.
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
IO_BEAM = 8
DIM = 768


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def prefix_fbin(src: Path, dst: Path, nrows: int):
    rows, dim = fbin_shape(src)
    if not (0 < nrows <= rows):
        raise ValueError("invalid prefix size")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated source fbin")
            fout.write(b)
            remaining -= len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    args = ap.parse_args()

    for attr in ("binary", "queries", "gt", "index_prefix", "work", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if dim != DIM or args.train_rows > rows:
        raise ValueError(f"expected >= {args.train_rows} x {DIM}, got {rows} x {dim}")

    qprefix = args.work / "train-prefix.fbin"
    prefix_fbin(args.queries, qprefix, args.train_rows)
    trace = args.out / "medoid-k1-train5000-trace.jsonl"
    output = args.out / "medoid-k1-train5000-output.json"
    inp = args.out / "medoid-k1-train5000-input.json"

    phase = {
        "queries": str(qprefix),
        "groundtruth": str(args.gt),
        "search_list": [1],
        "beam_width": IO_BEAM,
        "recall_at": 1,
        "num_threads": THREADS,
        "is_flat_search": False,
        "distance": "inner_product",
        "vector_filters_file": None,
        "num_nodes_to_cache": None,
        "search_io_limit": None,
        "post_processor": None,
    }
    cfg = {
        "search_directories": [str(args.work)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": str(args.index_prefix),
                },
                "search_phase": phase,
            },
        }],
    }
    save(inp, cfg)

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    env["DISKANN_TRACE_FILE"] = str(trace)
    for name in (
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (args.out / "benchmark.log").open("w") as log:
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
        raise ValueError("trace query order mismatch")

    trace_lens = np.asarray([len(r["ids"]) for r in records], dtype=np.int64)
    io_counts = np.asarray([int(r["io_count"]) for r in records], dtype=np.int64)
    if np.any(trace_lens <= 0):
        raise ValueError("empty traversal trace")

    summary = {
        "queries": args.train_rows,
        "search": {
            "k": 1,
            "search_l": 1,
            "beam_width": IO_BEAM,
            "threads": THREADS,
            "start": "DiskANN medoid",
            "metric": "inner_product",
        },
        "trace": {
            "mean_expanded_vertices": float(trace_lens.mean()),
            "median_expanded_vertices": float(np.median(trace_lens)),
            "p95_expanded_vertices": float(np.quantile(trace_lens, 0.95)),
            "mean_total_io": float(io_counts.mean()),
            "median_total_io": float(np.median(io_counts)),
            "p95_total_io": float(np.quantile(io_counts, 0.95)),
        },
        "trace_file": trace.name,
    }
    save(args.out / "trace-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
