#!/usr/bin/env python3
"""Order-balanced Catapult vs portal+Catapult SSD confirmation.

For each k and Catapult seed, run both A->B and B->A where:
  A = paper CatapultDB
  B = static portal + paper CatapultDB

The per-seed paired speed ratio is the geometric mean of the two directional
ratios. This cancels a multiplicative first/second-position timing effect and
makes the comparison robust to warm filesystem / SSD state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np

from qualify_medrag_zipf_four_arm import (
    CATAPULT_SEEDS,
    IO_BEAM,
    THREADS,
    fbin_shape,
    prefix_fbin,
    run_disk,
    save,
)

METHOD_A = "catapult"
METHOD_B = "portal-catapult"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--portal-router", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-values", default="1,2,4,8,16")
    args = ap.parse_args()

    for attr in ("binary", "queries", "gt", "index_prefix", "portal_router", "work", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    rows_total, dim = fbin_shape(args.queries)
    if rows_total != 10000 or dim != 768:
        raise ValueError(f"expected 10000x768 queries, got {rows_total}x{dim}")
    if not args.portal_router.is_file():
        raise FileNotFoundError(args.portal_router)
    os.environ["MEDRAG_ZIPF_PORTAL_ROUTER"] = str(args.portal_router)

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("runner exposes fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)

            for k in k_values:
                for seed in CATAPULT_SEEDS:
                    # Direction 0: A then B.
                    for direction, order in (
                        ("A-B", (METHOD_A, METHOD_B)),
                        ("B-A", (METHOD_B, METHOD_A)),
                    ):
                        for position, method in enumerate(order):
                            tag = f"k{k}-seed{seed}-{direction}-p{position}-{method}"
                            print(f"RUN {tag}", flush=True)
                            row = run_disk(
                                args.binary,
                                args.work,
                                args.out,
                                tag,
                                args.index_prefix,
                                args.queries,
                                args.gt,
                                k,
                                method,
                                seed,
                            )
                            rows.append({
                                "k": k,
                                "seed": seed,
                                "direction": direction,
                                "position": position,
                                "method": method,
                                **row,
                            })
                            save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)

    summary = {}
    for k in k_values:
        seed_rows = []
        for seed in CATAPULT_SEEDS:
            rr = [r for r in rows if r["k"] == k and r["seed"] == seed]
            by = {(r["direction"], r["method"]): r for r in rr}
            if len(by) != 4:
                raise ValueError(f"missing paired rows for k={k}, seed={seed}")

            a_first = by[("A-B", METHOD_A)]
            b_second = by[("A-B", METHOD_B)]
            b_first = by[("B-A", METHOD_B)]
            a_second = by[("B-A", METHOD_A)]

            ratio_ab = float(b_second["qps"]) / float(a_first["qps"])
            ratio_ba = float(b_first["qps"]) / float(a_second["qps"])
            paired_ratio = math.sqrt(ratio_ab * ratio_ba)

            a_ios = np.mean([float(a_first["mean_ios"]), float(a_second["mean_ios"])])
            b_ios = np.mean([float(b_first["mean_ios"]), float(b_second["mean_ios"])])
            io_reduction = 1.0 - b_ios / a_ios

            seed_rows.append({
                "seed": seed,
                "A_then_B_qps_ratio_B_over_A": ratio_ab,
                "B_then_A_qps_ratio_B_over_A": ratio_ba,
                "order_balanced_qps_ratio_B_over_A": paired_ratio,
                "catapult_mean_ios": float(a_ios),
                "combined_mean_ios": float(b_ios),
                "combined_io_reduction_vs_catapult": float(io_reduction),
            })

        ratios = np.asarray(
            [r["order_balanced_qps_ratio_B_over_A"] for r in seed_rows], dtype=np.float64
        )
        reductions = np.asarray(
            [r["combined_io_reduction_vs_catapult"] for r in seed_rows], dtype=np.float64
        )
        summary[str(k)] = {
            "seeds": seed_rows,
            "paired_qps_ratio_geomean": float(np.exp(np.log(ratios).mean())),
            "paired_qps_ratio_median": float(np.median(ratios)),
            "paired_qps_ratio_min": float(ratios.min()),
            "paired_qps_ratio_max": float(ratios.max()),
            "combined_io_reduction_mean": float(reductions.mean()),
            "combined_io_reduction_min": float(reductions.min()),
            "combined_io_reduction_max": float(reductions.max()),
        }

    result = {
        "workload": "reconstructed MedRAG-Zipf, 10k",
        "comparison": {
            "A": METHOD_A,
            "B": METHOD_B,
            "order_control": "for each k and seed, run A->B and B->A",
            "paired_ratio": "sqrt((B_after/A_before)*(B_before/A_after))",
        },
        "search": {
            "k_values": list(k_values),
            "catapult_seeds": list(CATAPULT_SEEDS),
            "threads": THREADS,
            "ssd_io_beam_width": IO_BEAM,
            "speed_device_lock": str(lock_path),
        },
        "summary": summary,
    }
    save(args.out / "catapult-vs-combined-paired.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
