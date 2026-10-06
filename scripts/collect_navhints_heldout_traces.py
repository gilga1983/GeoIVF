#!/usr/bin/env python3
"""Collect canonical 16K NavHints traversal traces on a held-out query range."""
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
LS = (40, 80, 160, 320)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin")
    return rows, dim


def range_fbin(src: Path, dst: Path, start: int, count: int):
    rows, dim = fbin_shape(src)
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError("invalid query range")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", count, dim))
        remain = count * dim * 4
        while remain:
            b = fin.read(min(16 << 20, remain))
            if not b:
                raise ValueError("truncated source")
            fout.write(b)
            remain -= len(b)


def fake_gt(path: Path, rows: int):
    with path.open("wb") as f:
        f.write(struct.pack("<II", rows, 1))
        np.zeros(rows, dtype="<u4").tofile(f)


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "index-prefix", "start-ivf", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--start-row", type=int, default=9000)
    ap.add_argument("--rows", type=int, default=1000)
    args = ap.parse_args()
    for name in ("binary", "queries", "index_prefix", "start_ivf", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    total, dim = fbin_shape(args.queries)
    if dim != DIM or args.start_row + args.rows > total:
        raise ValueError("unexpected workload range")

    qrange = args.work / "heldout-range.fbin"
    gt = args.work / "skip-recall.gt"
    range_fbin(args.queries, qrange, args.start_row, args.rows)
    fake_gt(gt, args.rows)

    trace_base = args.out / "heldout.jsonl"
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
                    "queries": str(qrange),
                    "groundtruth": str(gt),
                    "search_list": list(LS),
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
    env["DISKANN_TRACE_FILE"] = str(trace_base)
    env["DISKANN_HINT_IVF_FILE"] = str(args.start_ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    for name in (
        "DISKANN_REGIONAL_HINT_FILE", "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_STAGE_HINT_IVF_FILE", "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS", "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE", "DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE", "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE", "DISKANN_VERTEX_HINT_VARIANT",
        "DISKANN_EMBEDDED_HINTS",
    ):
        env.pop(name, None)

    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    with lockp.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (args.out / "benchmark.log").open("w") as log:
            subprocess.run(
                [str(args.binary), "run", "--input-file", str(inp), "--output-file", str(output)],
                stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
            )

    summary = {
        "query_start": args.start_row,
        "queries": args.rows,
        "Ls": list(LS),
        "beam": BEAM,
        "threads": THREADS,
        "policy": "canonical 16K optimized Hint-IVF start-only, nprobe=8",
        "traces": {},
    }
    for l in LS:
        path = args.out / f"heldout.L{l}.jsonl"
        records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if len(records) != args.rows:
            raise ValueError(f"L{l}: expected {args.rows} traces, got {len(records)}")
        lens = np.asarray([len(r["ids"]) for r in records], dtype=np.int64)
        summary["traces"][str(l)] = {
            "file": path.name,
            "mean_expanded": float(lens.mean()),
            "median_expanded": float(np.median(lens)),
            "p95_expanded": float(np.quantile(lens, .95)),
        }
    save(args.out / "trace-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
