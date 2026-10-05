#!/usr/bin/env python3
"""Compare canonical, off-policy staged, and on-policy staged NavHints."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from qualify_staged_navhints import (
    BEAM,
    K,
    LS,
    THREADS,
    aggregate,
    fbin_shape,
    interp,
    payload,
    run_one,
    save,
    suffix_fbin,
)


def main():
    ap = argparse.ArgumentParser()
    for name in (
        "binary", "queries", "gt", "index-prefix", "ivf-16k", "ivf-start8k",
        "ivf-stage8k-offpolicy", "ivf-stage8k-onpolicy", "work", "out",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in (
        "binary", "queries", "gt", "index_prefix", "ivf_16k", "ivf_start8k",
        "ivf_stage8k_offpolicy", "ivf_stage8k_onpolicy", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    methods = ("start16k", "start8k", "staged-offpolicy", "staged-onpolicy")
    runs = {m: [] for m in methods}

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

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
                print(f"rep {rep}: {' '.join(order)}", flush=True)
                for method in order:
                    if method == "start16k":
                        kw = dict(start_ivf=args.ivf_16k, start_probe=8)
                    elif method == "start8k":
                        kw = dict(start_ivf=args.ivf_start8k, start_probe=4)
                    elif method == "staged-offpolicy":
                        kw = dict(
                            start_ivf=args.ivf_start8k,
                            start_probe=4,
                            stage_ivf=args.ivf_stage8k_offpolicy,
                            stage_probe=4,
                            stage_hops=8,
                        )
                    else:
                        kw = dict(
                            start_ivf=args.ivf_start8k,
                            start_probe=4,
                            stage_ivf=args.ivf_stage8k_onpolicy,
                            stage_probe=4,
                            stage_hops=8,
                        )
                    rr = run_one(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        held,
                        args.gt,
                        args.index_prefix,
                        **kw,
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    print(
                        f"finished rep={rep} method={method} "
                        f"L10 recall={float(rr[0]['recall']):.3f} ios={float(rr[0]['mean_ios']):.3f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}
    memory = {
        "start16k": payload(args.ivf_16k),
        "start8k": payload(args.ivf_start8k),
        "staged-offpolicy": payload(args.ivf_start8k) + payload(args.ivf_stage8k_offpolicy),
        "staged-onpolicy": payload(args.ivf_start8k) + payload(args.ivf_stage8k_onpolicy),
    }

    fixed = {}
    common_lo = max(summary[m]["10"]["recall_percent"] for m in methods)
    common_hi = min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0, 45.0, 55.0, 65.0):
        if not (common_lo <= target <= common_hi):
            continue
        item = {}
        for method in methods:
            io = interp(summary[method], target, "mean_ios")
            lat = interp(summary[method], target, "median_latency_us")
            item[method] = {
                "mean_ios": io,
                "latency_us": lat,
                "runtime_payload_bytes": memory[method],
            }
        for lhs, rhs, label in (
            ("staged-onpolicy", "start16k", "onpolicy_vs_start16k"),
            ("staged-onpolicy", "staged-offpolicy", "onpolicy_vs_offpolicy"),
        ):
            item[label] = {
                "io_change_percent": 100.0 * (
                    item[lhs]["mean_ios"] / item[rhs]["mean_ios"] - 1.0
                ),
                "latency_change_percent": 100.0 * (
                    item[lhs]["latency_us"] / item[rhs]["latency_us"] - 1.0
                ),
            }
        fixed[str(target)] = item

    same_l = {}
    for l in LS:
        same_l[str(l)] = {}
        for method in methods:
            row = summary[method][str(l)]
            same_l[str(l)][method] = {
                "recall_percent": row["recall_percent"],
                "mean_ios": row["mean_ios"],
                "latency_us": row["median_latency_us"],
                "qps": row["median_qps"],
            }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "training": {
            "start_policy": "8K stage-0 vocabulary learned from ordinary medoid-start teacher L=4 traces",
            "offpolicy_stage8": "residual stage-8 score from ordinary medoid-start teacher L=4 traces",
            "onpolicy_stage8": "residual stage-8 score from 8K-NavHints-start teacher L=4 traces",
            "score": "S_8(v)=sum max(first_expansion_position-8,0)",
        },
        "arms": {
            "start16k": "16K hints at query start, 512 cells/probe 8",
            "start8k": "8K hints at query start, 256 cells/probe 4",
            "staged-offpolicy": "8K start + off-policy 8K at hop 8",
            "staged-onpolicy": "8K start + on-policy 8K at hop 8",
        },
        "memory_payload_bytes": memory,
        "evaluation": {
            "K": K,
            "Ls": list(LS),
            "beam": BEAM,
            "threads": THREADS,
            "repetitions": args.reps,
        },
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
    }
    save(args.out / "onpolicy-staged-navhints.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
