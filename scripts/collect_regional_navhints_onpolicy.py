#!/usr/bin/env python3
"""Collect long on-policy traces from deployed 16K start-only NavHints."""
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
TEACHER_L = 64


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
    if not 0 < nrows <= rows:
        raise ValueError("invalid prefix")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated source")
            fout.write(b)
            remaining -= len(b)


def fake_gt(path: Path, rows: int):
    with path.open("wb") as f:
        f.write(struct.pack("<II", rows, 1))
        np.zeros(rows, dtype="<u4").tofile(f)


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "index-prefix", "start-ivf", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=9000)
    args = ap.parse_args()

    for name in ("binary", "queries", "index_prefix", "start_ivf", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if dim != DIM or not 0 < args.train_rows <= rows:
        raise ValueError("unexpected workload shape")

    qprefix = args.work / "train-prefix.fbin"
    gt = args.work / "skip-recall.gt"
    prefix_fbin(args.queries, qprefix, args.train_rows)
    fake_gt(gt, args.train_rows)

    trace = args.out / "navhints16k-teacher-L64.jsonl"
    inp = args.out / "trace.input.json"
    output = args.out / "trace.output.json"

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
                    "groundtruth": str(gt),
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
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    for name in (
        "DISKANN_REGIONAL_HINT_FILE",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)

    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    with lockp.open("w") as lock:
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
        raise ValueError("trace query numbering mismatch")
    lens = np.asarray([len(r["ids"]) for r in records], dtype=np.int64)
    ios = np.asarray([int(r["io_count"]) for r in records], dtype=np.int64)
    summary = {
        "queries": args.train_rows,
        "policy": "canonical 16K optimized Hint-IVF start-only, nprobe=8",
        "teacher": {"L": TEACHER_L, "beam": BEAM, "threads": THREADS},
        "mean_expanded_vertices": float(lens.mean()),
        "median_expanded_vertices": float(np.median(lens)),
        "p95_expanded_vertices": float(np.quantile(lens, .95)),
        "mean_io": float(ios.mean()),
        "median_io": float(np.median(ios)),
        "p95_io": float(np.quantile(ios, .95)),
        "trace_file": trace.name,
    }
    save(args.out / "trace-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
