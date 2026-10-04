#!/usr/bin/env python3
"""Held-out evaluation for navigation-heavy-hitter cache variants."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from qualify_waypoint_cache_heldout import (
    THREADS,
    fbin_shape,
    suffix_fbin,
    run_one,
    save,
)

DEFAULT_METHODS = (
    "heavy-local-freq-x4",
    "heavy-local-freq-x6",
    "heavy-local-freq-x8",
    "heavy-local-freq-x10",
    "heavy-local-freq-x20",
    "heavy-local-freq-x40",
    "heavy-local-skip-x4",
    "heavy-local-skip-x6",
    "heavy-local-skip-x8",
    "heavy-local-skip-x10",
    "heavy-local-skip-x20",
    "heavy-local-skip-x40",
    "heavy-global-freq-b2000",
    "heavy-global-freq-b2500",
    "heavy-global-freq-b2800",
    "heavy-global-skip-b2000",
    "heavy-global-skip-b2500",
    "heavy-global-skip-b2800",
)


def median(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--portal-router", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for attr in (
        "binary", "queries", "gt", "index_prefix",
        "portal_router", "cache_dir", "work", "out",
    ):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != 10000 or dim != 768 or args.train_rows != 5000:
        raise ValueError(f"expected frozen 10000x768 split 5000/5000, got {rows}x{dim}")

    heldout = args.work / "heldout-5000.fbin"
    nheld, _ = suffix_fbin(args.queries, heldout, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    cache_map = {"portal": None}
    for name in DEFAULT_METHODS:
        p = args.cache_dir / f"{name}.bin"
        if not p.is_file():
            raise FileNotFoundError(p)
        cache_map[name] = p

    methods = tuple(cache_map.keys())
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    all_rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    tag = f"r{rep}-{method}"
                    print(f"RUN {tag}", flush=True)
                    row = run_one(
                        args.binary,
                        args.work,
                        args.out,
                        tag,
                        heldout,
                        args.gt,
                        args.index_prefix,
                        args.portal_router,
                        cache_map[method],
                    )
                    all_rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)

    summary = {}
    base_rows = [r for r in all_rows if r["method"] == "portal"]
    base_ios = mean(base_rows, "mean_ios")
    base_qps = median(base_rows, "qps")
    for method in methods:
        rr = [r for r in all_rows if r["method"] == method]
        obj = {
            "runs": len(rr),
            "mean_ios": mean(rr, "mean_ios"),
            "median_qps": median(rr, "qps"),
            "median_latency_us": median(rr, "mean_latency"),
            "median_io_time_us": median(rr, "mean_io_time"),
            "median_cpu_time_us": median(rr, "mean_cpu_time"),
        }
        obj["io_reduction_vs_portal"] = 1.0 - obj["mean_ios"] / base_ios
        obj["qps_speedup_vs_portal"] = obj["median_qps"] / base_qps
        summary[method] = obj

    ranked = sorted(
        (
            (name, vals["mean_ios"], vals["median_qps"])
            for name, vals in summary.items()
            if name != "portal"
        ),
        key=lambda x: (x[1], -x[2]),
    )

    result = {
        "workload": "reconstructed MedRAG-Zipf heldout suffix",
        "split": {"train_prefix": 5000, "heldout_suffix": 5000},
        "search": {
            "k": 1,
            "search_l": 1,
            "beam_width": 8,
            "threads": THREADS,
            "repetitions": args.reps,
            "recall": "skipped in navigation-heavy-hitter screening",
        },
        "summary": summary,
        "ranked_by_io": [
            {"method": n, "mean_ios": io, "median_qps": qps}
            for n, io, qps in ranked
        ],
    }
    save(args.out / "navigation-heavy-hitters-heldout.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
