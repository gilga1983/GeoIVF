#!/usr/bin/env python3
"""Collect ordinary medoid-start DiskANN traversal traces at several teacher L values.

The first 5000 workload queries are training only. One DiskANN invocation
evaluates all requested search-list sizes and the existing trace patch writes
one JSONL file per L. These traces are later distilled into global ID-only
landmark dictionaries, while deployment remains fixed at cheap online L=1.
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

def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
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


def trace_path(base: Path, lval: int) -> Path:
    # patch_diskann_waypoint_trace.py transforms foo.jsonl -> foo.L4.jsonl
    return base.with_suffix(f".L{lval}{base.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--teacher-ls", default="1,4,8,16,32,64")
    args = ap.parse_args()

    for attr in ("binary", "queries", "gt", "index_prefix", "work", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    ls = [int(x) for x in args.teacher_ls.split(",") if x.strip()]
    if not ls or len(set(ls)) != len(ls) or any(x <= 0 for x in ls):
        raise ValueError("teacher L values must be unique positive integers")

    rows, dim = fbin_shape(args.queries)
    if args.train_rows > rows:
        raise ValueError(f"expected at least {args.train_rows} queries, got {rows}")
    if dim <= 0:
        raise ValueError("invalid query dimension")

    qprefix = args.work / "train-prefix.fbin"
    prefix_fbin(args.queries, qprefix, args.train_rows)
    trace_base = args.out / "medoid-teacher.jsonl"
    output = args.out / "medoid-teacher-output.json"
    inp = args.out / "medoid-teacher-input.json"

    phase = {
        "queries": str(qprefix),
        "groundtruth": str(args.gt),
        "search_list": ls,
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
    env["DISKANN_TRACE_FILE"] = str(trace_base)
    for name in (
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_GLOBAL_START_IDS_FILE",
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

    summary = {"queries": args.train_rows, "teacher_ls": ls, "traces": {}}
    for lval in ls:
        path = trace_path(trace_base, lval)
        if not path.is_file():
            raise FileNotFoundError(path)
        records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if len(records) != args.train_rows:
            raise ValueError(f"L={lval}: expected {args.train_rows} traces, got {len(records)}")
        if [int(r["query"]) for r in records] != list(range(args.train_rows)):
            raise ValueError(f"L={lval}: trace query numbering mismatch")
        lens = np.asarray([len(r["ids"]) for r in records], dtype=np.int64)
        ios = np.asarray([int(r["io_count"]) for r in records], dtype=np.int64)
        if np.any(lens <= 0):
            raise ValueError(f"L={lval}: empty traversal trace")
        summary["traces"][str(lval)] = {
            "file": path.name,
            "mean_expanded_vertices": float(lens.mean()),
            "median_expanded_vertices": float(np.median(lens)),
            "p95_expanded_vertices": float(np.quantile(lens, 0.95)),
            "mean_io": float(ios.mean()),
            "median_io": float(np.median(ios)),
            "p95_io": float(np.quantile(ios, 0.95)),
        }

    save(args.out / "multi-l-trace-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
