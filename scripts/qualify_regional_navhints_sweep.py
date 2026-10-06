#!/usr/bin/env python3
"""Runtime sweep over regional-map resolution and equal per-region hint budget."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from qualify_regional_navhints_runtime import (
    K, LS, THREADS,
    aggregate, fbin_shape, interp, payload_ivf, run_one, save, suffix_fbin, suffix_gt,
)


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "gt5000", "index-prefix", "ivf-16k", "regional-root", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt5000", "index_prefix", "ivf_16k", "regional_root", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected workload")

    held = args.work / "heldout-last1000.fbin"
    gt = args.work / "heldout-last1000.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

    variants = [(r, h) for r in (128, 256, 512) for h in (8, 16)]
    paths = {
        f"regional{r}x{h}": args.regional_root / f"regional-{r}x{h}.bin"
        for r, h in variants
    }
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    methods = ("canonical16k",) + tuple(paths)
    runs = {m: [] for m in methods}

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for method in order:
                    rr = run_one(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        held,
                        gt,
                        args.index_prefix,
                        args.ivf_16k,
                        None if method == "canonical16k" else paths[method],
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    row = rr[2]  # L=40
                    print(
                        f"finished rep={rep} method={method} "
                        f"L40 recall={float(row['recall']):.3f} "
                        f"io={float(row['mean_ios']):.3f} "
                        f"cpu={float(row['mean_cpu_time']):.1f} "
                        f"lat={float(row['mean_latency']):.1f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}
    canonical = summary["canonical16k"]

    same_l = {}
    for l in LS:
        base = canonical[str(l)]
        item = {}
        for method in methods:
            row = summary[method][str(l)]
            item[method] = {
                "recall_percent": row["recall_percent"],
                "mean_ios": row["mean_ios"],
                "latency_us": row["median_latency_us"],
                "cpu_us": row["median_cpu_us"],
                "mean_comparisons": row["mean_comparisons"],
                "recall_delta_points_vs_canonical": row["recall_percent"] - base["recall_percent"],
                "io_change_percent_vs_canonical": 100.0 * (row["mean_ios"] / base["mean_ios"] - 1.0),
                "latency_change_percent_vs_canonical": 100.0 * (
                    row["median_latency_us"] / base["median_latency_us"] - 1.0
                ),
                "cpu_change_percent_vs_canonical": 100.0 * (
                    row["median_cpu_us"] / base["median_cpu_us"] - 1.0
                ),
            }
        same_l[str(l)] = item

    fixed = {}
    lo = max(summary[m]["10"]["recall_percent"] for m in methods)
    hi = min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0, 45.0, 55.0, 65.0):
        if not lo <= target <= hi:
            continue
        base = {
            "mean_ios": interp(canonical, target, "mean_ios"),
            "latency_us": interp(canonical, target, "median_latency_us"),
            "cpu_us": interp(canonical, target, "median_cpu_us"),
            "mean_comparisons": interp(canonical, target, "mean_comparisons"),
        }
        arms = {}
        for method in methods:
            cur = {
                "mean_ios": interp(summary[method], target, "mean_ios"),
                "latency_us": interp(summary[method], target, "median_latency_us"),
                "cpu_us": interp(summary[method], target, "median_cpu_us"),
                "mean_comparisons": interp(summary[method], target, "mean_comparisons"),
            }
            cur["io_change_percent_vs_canonical"] = 100.0 * (
                cur["mean_ios"] / base["mean_ios"] - 1.0
            )
            cur["latency_change_percent_vs_canonical"] = 100.0 * (
                cur["latency_us"] / base["latency_us"] - 1.0
            )
            cur["cpu_change_percent_vs_canonical"] = 100.0 * (
                cur["cpu_us"] / base["cpu_us"] - 1.0
            )
            arms[method] = cur
        fixed[str(target)] = arms

    memory = {
        "canonical_hint_ivf": payload_ivf(args.ivf_16k),
        **{m: p.stat().st_size for m, p in paths.items()},
    }

    # Compact score: average fixed-recall latency change over available targets.
    ranking = []
    for method in methods[1:]:
        vals = [
            fixed[t][method]["latency_change_percent_vs_canonical"]
            for t in fixed
        ]
        io = [
            fixed[t][method]["io_change_percent_vs_canonical"]
            for t in fixed
        ]
        ranking.append({
            "method": method,
            "mean_fixed_recall_latency_change_percent": sum(vals) / len(vals),
            "mean_fixed_recall_io_change_percent": sum(io) / len(io),
        })
    ranking.sort(key=lambda x: x["mean_fixed_recall_latency_change_percent"])

    result = {
        "workload": "MedRAG-Zipf final 1000 queries; exact PubMed1M IP top-16 truth",
        "training": "first 9000 queries, on-policy canonical 16K NavHints, teacher L=64",
        "sweep": {
            "regions": [128, 256, 512],
            "hints_per_region": [8, 16],
            "warmup_entry_position": 8,
            "runtime": "score at most one newly entered region per natural beam; offer one best unvisited hint through fixed-L gate",
        },
        "memory_payload_bytes": memory,
        "evaluation": {
            "K": K,
            "Ls": list(LS),
            "threads": THREADS,
            "repetitions": args.reps,
        },
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
        "ranking_by_mean_fixed_recall_latency": ranking,
    }
    save(args.out / "regional-navhints-runtime-sweep.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
