#!/usr/bin/env python3
"""Collect on-policy traces from the deployed 8K start-only NavHints policy."""
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
DIM = 768
TEACHER_L = 4


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def prefix_fbin(src: Path, dst: Path, nrows: int):
    rows, dim = fbin_shape(src)
    if not (0 < nrows <= rows):
        raise ValueError("invalid prefix")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * 4
        while remaining:
            block = fin.read(min(16 << 20, remaining))
            if not block:
                raise ValueError("truncated fbin")
            fout.write(block)
            remaining -= len(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--start-ivf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "start_ivf", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if dim != DIM or args.train_rows > rows:
        raise ValueError("unexpected workload shape")
    if not args.start_ivf.is_file():
        raise FileNotFoundError(args.start_ivf)

    qprefix = args.work / "train-prefix.fbin"
    prefix_fbin(args.queries, qprefix, args.train_rows)
    trace = args.out / "navhints8k-teacher-L4.jsonl"
    inp = args.out / "navhints8k-teacher-L4.input.json"
    output = args.out / "navhints8k-teacher-L4.output.json"

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
                "search_phase": {
                    "queries": str(qprefix),
                    "groundtruth": str(args.gt),
                    "search_list": [TEACHER_L],
                    "beam_width": BEAM,
                    "recall_at": 1,
                    "num_threads": THREADS,
                    "is_flat_search": False,
                    "distance": "inner_product",
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
    env["DISKANN_HINT_IVF_FILE"] = str(args.start_ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "4"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    for name in (
        "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        print(f"waiting for speed-device lock: {lock_path}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("acquired speed-device lock", flush=True)
        with (args.out / "trace-benchmark.log").open("w") as log:
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
    if np.any(lens <= 0):
        raise ValueError("empty trace")

    stage8_eligible = sum(1 for r in records if len(r["ids"]) > 8)
    summary = {
        "queries": args.train_rows,
        "policy": "8K start-only NavHints, 256 cells, probe 4",
        "teacher": {"L": TEACHER_L, "beam": BEAM, "threads": THREADS},
        "trace": {
            "mean_expanded_vertices": float(lens.mean()),
            "median_expanded_vertices": float(np.median(lens)),
            "p95_expanded_vertices": float(np.quantile(lens, 0.95)),
            "mean_io": float(ios.mean()),
            "median_io": float(np.median(ios)),
            "p95_io": float(np.quantile(ios, 0.95)),
            "queries_with_position_gt_8": stage8_eligible,
            "fraction_with_position_gt_8": stage8_eligible / args.train_rows,
        },
        "trace_file": trace.name,
    }
    save(args.out / "onpolicy-trace-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
