#!/usr/bin/env python3
"""Evaluate prepared NavHints scoring/training ablation states.

Every state directory contains:
  ivf.bin
  state.json

The search implementation, nlist=512, nprobe=8, DiskANN index, beam, and online
L are identical across states. We report both the L=1 Recall@1 diagnostic and
a realistic Recall@10 point at L=10.
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
SEARCHES = (("r1-l1", 1, 1), ("r10-l10", 10, 10))


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("bad vector header")
    n, d = struct.unpack("<II", raw)
    return n, d


def slice_fbin(src: Path, dst: Path, start: int, count: int):
    rows, dim = shape(src)
    if start + count > rows:
        raise ValueError("query slice outside source")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", count, dim))
        remaining = count * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)


def result_rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            out.append(obj)
        else:
            for v in obj.values():
                out.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(result_rows(v))
    return out


def run_one(binary, out, tag, queries, gt, index_prefix, ivf, recall_at, lval):
    cfg = {
        "search_directories": [str(out)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": str(index_prefix),
                },
                "search_phase": {
                    "queries": str(queries),
                    "groundtruth": str(gt),
                    "search_list": [lval],
                    "beam_width": BEAM,
                    "recall_at": recall_at,
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
    inp = out / f"{tag}.input.json"
    output = out / f"{tag}.output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rr = result_rows(json.loads(output.read_text()))
    if len(rr) != 1:
        raise ValueError(f"{tag}: expected one result row")
    return dict(rr[0])


def avg(rows, key):
    return float(np.mean([float(x[key]) for x in rows]))


def med(rows, key):
    return float(np.median([float(x[key]) for x in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--state-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for n in ("binary", "queries", "gt", "index_prefix", "state_root", "work", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    qrows, qdim = shape(args.queries)
    if (qrows, qdim) != (10000, 768):
        raise ValueError("unexpected PubMed query set")
    held = args.work / "heldout.fbin"
    slice_fbin(args.queries, held, 5000, 5000)

    states = []
    for meta_path in sorted(args.state_root.rglob("state.json")):
        meta = json.loads(meta_path.read_text())
        ivf = meta_path.parent / "ivf.bin"
        if not ivf.is_file():
            raise FileNotFoundError(ivf)
        states.append((meta["name"], ivf, meta))
    if not states:
        raise ValueError("no prepared states")
    if len({x[0] for x in states}) != len(states):
        raise ValueError("duplicate state names")

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for search_name, recall_at, lval in SEARCHES:
                methods = [x[0] for x in states]
                state_by_name = {x[0]: x for x in states}
                for rep in range(args.reps):
                    shift = rep % len(methods)
                    order = methods[shift:] + methods[:shift]
                    for name in order:
                        _, ivf, meta = state_by_name[name]
                        row = run_one(
                            args.binary,
                            args.out,
                            f"{search_name}-r{rep}-{name}",
                            held,
                            args.gt,
                            args.index_prefix,
                            ivf,
                            recall_at,
                            lval,
                        )
                        rows.append({
                            "search": search_name,
                            "rep": rep,
                            "state": name,
                            **meta,
                            **row,
                        })
                        save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for search_name, _, _ in SEARCHES:
        summary[search_name] = {}
        sr = [x for x in rows if x["search"] == search_name]
        for name, _, meta in states:
            rr = [x for x in sr if x["state"] == name]
            summary[search_name][name] = {
                **meta,
                "rounds": len(rr),
                "recall_percent": avg(rr, "recall"),
                "mean_ios": avg(rr, "mean_ios"),
                "median_qps": med(rr, "qps"),
                "median_latency_us": med(rr, "mean_latency"),
                "median_cpu_us": med(rr, "mean_cpu_time"),
            }

    result = {
        "workload": "PubMed1M / MedRAG-Zipf frozen final 5000 queries",
        "router": {"nlist": 512, "nprobe": 8},
        "implementation": "packed-direct coarse PQ + pooled vectorized fine scoring",
        "summary": summary,
    }
    save(args.out / "paper-ablation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
