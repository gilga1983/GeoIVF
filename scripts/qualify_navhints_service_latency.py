#!/usr/bin/env python3
"""End-to-end, serial, causal PubMed SSD evaluation at matched Recall@10.

Uses identical pinned graph/PQ, query order and 4K warmup / 1K measured
suffix for baseline, junction-only, selected NavHints Core, and author-aligned
Catapult routing. The on-request Rust scope includes learning/feedback; run_one's
native mean_latency and qps are NOT used for complete-service conclusions.
No heavy work occurs merely by importing or running --help.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

from qualify_navhints_catapult_headtohead import (
    K, BEAM, SEARCH_THREADS, TRAIN, WARM, MEASURE, DIM,
    HASH_BITS, CATAPULT_CAPACITY, run_one, shape, slice_fbin, snapshot_entries,
)

EXPECTED_FIELDS = (
    "query_index", "service_us", "engine_us", "remainder_us", "io_us",
    "pq_preprocess_us", "reported_cpu_us", "reads", "comparisons", "hops",
)
PERCENTILES = (50, 95, 99)
CORE = {"junctions": 14464, "recent": 2048, "starts": 8, "nlist": 512, "nprobe": 8}


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def percentile(samples: list[float], pct: int) -> float:
    data = sorted(samples)
    if not data:
        raise ValueError("no latency samples")
    position = (len(data) - 1) * pct / 100.0
    lo = int(position)
    hi = min(len(data) - 1, lo + 1)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def profile_summary(path: Path, warmup: int = WARM,
                    measured: int = MEASURE) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"native service profile missing: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != measured:
        raise ValueError(f"{path}: expected {measured} queries, got {len(rows)}")
    if tuple(rows[0]) != EXPECTED_FIELDS:
        raise ValueError(f"{path}: unexpected native service schema {list(rows[0])}")
    data = []
    for offset, row in enumerate(rows):
        vals = {k: int(row[k]) for k in EXPECTED_FIELDS}
        if vals["query_index"] != warmup + offset:
            raise ValueError(f"{path}: out-of-order query index at offset {offset}")
        if vals["service_us"] <= 0 or vals["engine_us"] <= 0:
            raise ValueError(f"{path}: nonpositive service or engine time")
        if vals["service_us"] != vals["engine_us"] + vals["remainder_us"]:
            raise ValueError(f"{path}: incomplete service accounting")
        if min(vals[k] for k in EXPECTED_FIELDS if k != "query_index") < 0:
            raise ValueError(f"{path}: negative counter")
        data.append(vals)
    total = sum(r["service_us"] for r in data)
    out = {
        "measured_queries": measured,
        "mean_service_us": statistics.mean(r["service_us"] for r in data),
        "mean_engine_us": statistics.mean(r["engine_us"] for r in data),
        "mean_remainder_us": statistics.mean(r["remainder_us"] for r in data),
        "mean_io_us": statistics.mean(r["io_us"] for r in data),
        "mean_pq_preprocess_us": statistics.mean(r["pq_preprocess_us"] for r in data),
        "mean_reads": statistics.mean(r["reads"] for r in data),
        "mean_comparisons": statistics.mean(r["comparisons"] for r in data),
        "mean_hops": statistics.mean(r["hops"] for r in data),
        "serial_throughput_qps": 1e6 * measured / total,
        "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for p in PERCENTILES:
        out[f"service_p{p}_us"] = percentile([float(r["service_us"]) for r in data], p)
    return out


def frontier(points: dict[int, dict]) -> list[dict]:
    sorted_points = sorted(points.values(), key=lambda p: (p["recall"], p["mean_service_us"]))
    ret = []
    for point in sorted_points:
        if ret and point["recall"] <= ret[-1]["recall"] + 1e-8:
            if point["mean_service_us"] < ret[-1]["mean_service_us"]:
                ret[-1] = point
        else:
            ret.append(point)
    return ret


def interpolate(points: dict[int, dict], recall: float) -> dict | None:
    ordered = frontier(points)
    if not ordered or recall < ordered[0]["recall"] - 1e-8 or recall > ordered[-1]["recall"] + 1e-8:
        return None
    fields = ("mean_service_us", "mean_engine_us", "mean_remainder_us", "mean_io_us", "mean_reads")
    for left in ordered:
        if abs(recall - left["recall"]) <= 1e-8:
            return {key: left[key] for key in fields}
    for left, right in zip(ordered, ordered[1:]):
        if left["recall"] < recall < right["recall"]:
            frac = (recall - left["recall"]) / (right["recall"] - left["recall"])
            return {key: left[key] + frac * (right[key] - left[key]) for key in fields}
    return None


def matched_recall(measurements: dict, widths: list[int], reps: int,
                   seeds: list[int]) -> list[dict]:
    curves = {}
    for rep in range(reps):
        for mode in ("baseline", "junctions", "navhints"):
            curves[(rep, mode)] = {l: measurements[f"r{rep}-L{l}-{mode}"] for l in widths}
        for seed in seeds:
            curves[(rep, f"catapult-s{seed}")] = {
                l: measurements[f"r{rep}-L{l}-catapult-s{seed}"] for l in widths
            }
    rows = []
    for l in widths:
        target = statistics.mean(curves[(rep, "baseline")][l]["recall"] for rep in range(reps))
        result = {"baseline_L": l, "recall_at_10": target, "available": False}
        matched = {}
        for mode in ("baseline", "junctions", "navhints", "catapult"):
            variants = (
                [f"catapult-s{seed}" for seed in seeds] if mode == "catapult" else [mode]
            )
            pts = [interpolate(curves[(rep, variant)], target)
                   for rep in range(reps) for variant in variants]
            if any(p is None for p in pts):
                result["reason"] = f"{mode} lacks supported recall at target"
                break
            matched[mode] = {
                k: statistics.mean(p[k] for p in pts)
                for k in ("mean_service_us", "mean_engine_us", "mean_remainder_us",
                          "mean_io_us", "mean_reads")
            }
        if len(matched) == 4:
            result.update({"available": True, "arms": matched})
            for other in ("baseline", "junctions", "catapult"):
                result[f"navhints_latency_saved_vs_{other}_percent"] = 100 * (
                    1 - matched["navhints"]["mean_service_us"] / matched[other]["mean_service_us"]
                )
                result[f"navhints_reads_saved_vs_{other}_percent"] = 100 * (
                    1 - matched["navhints"]["mean_reads"] / matched[other]["mean_reads"]
                )
        rows.append(result)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("nav-bin", "cat-bin", "queries", "train-gt", "heldout-gt",
                 "index-prefix", "ivf", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--widths", default="40,80,160",
                    help="Comma-separated L values; smoke default deliberately small")
    ap.add_argument("--reps", type=int, default=1,
                    help="Use three rotated reps for a manuscript result")
    ap.add_argument("--cat-seeds", default="0",
                    help="Use 0,1,2 for a manuscript result")
    args = ap.parse_args()
    widths = sorted({int(x) for x in args.widths.split(",")})
    seeds = sorted({int(x) for x in args.cat_seeds.split(",")})
    if not widths or any(l < K for l in widths):
        raise ValueError("L must be at least K")
    if args.reps < 1 or args.reps > 5 or any(seed not in range(3) for seed in seeds):
        raise ValueError("unsupported repetition or Catapult seed")
    for name in ("nav_bin", "cat_bin", "queries", "train_gt", "heldout_gt", "ivf"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    if shape(args.queries) != (TRAIN + WARM + MEASURE, DIM):
        raise ValueError("unexpected PubMed 10K x 768 frozen source")
    manifest_path = args.ivf.with_suffix(args.ivf.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest["landmark_ids"]) != CORE["junctions"] or int(manifest["nlist"]) != CORE["nlist"]:
        raise RuntimeError("wrong selected NavHints directory")
    if args.ivf.stat().st_size + 512 * (64 + 4) + CORE["recent"] * 4 != 102940:
        raise RuntimeError("NavHints routing accounted payload changed")

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    train = slice_fbin(args.queries, args.work / "train5000.fbin", 0, TRAIN)
    held = slice_fbin(args.queries, args.work / "heldout5000.fbin", TRAIN, WARM + MEASURE)

    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < SEARCH_THREADS:
        raise RuntimeError("need at least four CPUs")
    os.sched_setaffinity(0, set(cpus[:SEARCH_THREADS]))
    measurements = {}
    snapshots = {}
    try:
        for L in widths:
            for seed in seeds:
                label = f"catapult-train-L{L}-s{seed}"
                snap = args.work / f"{label}.snapshot"
                run_one(args.cat_bin, label, train, args.train_gt, args.index_prefix, L,
                        args.out, mode="catapult", seed=seed,
                        snapshot_dump=snap, train=True)
                residents = snapshot_entries(snap)
                if not 0 < residents <= (1 << HASH_BITS) * CATAPULT_CAPACITY:
                    raise RuntimeError("Catapult snapshot occupancy incorrect")
                snapshots[label] = {"residents": residents,
                                    "sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}
                save(args.out / "snapshots.partial.json", snapshots)

        for rep in range(args.reps):
            order_widths = widths[rep % len(widths):] + widths[:rep % len(widths)]
            for j, L in enumerate(order_widths):
                arms = ["baseline", "junctions", "navhints"] + [
                    f"catapult-s{seed}" for seed in seeds
                ]
                shift = (j + rep) % len(arms)
                arms = arms[shift:] + arms[:shift]
                print(f"SERVICE_REP={rep} L={L} order={','.join(arms)}", flush=True)
                for arm in arms:
                    label = f"r{rep}-L{L}-{arm}"
                    profile = args.out / "profiles" / label
                    if arm.startswith("catapult-"):
                        seed = int(arm.rsplit("s", 1)[-1])
                        result = run_one(
                            args.cat_bin, label, held, args.heldout_gt, args.index_prefix,
                            L, args.out, mode="catapult", seed=seed,
                            snapshot_load=args.work / f"catapult-train-L{L}-s{seed}.snapshot",
                            extra_env={"DISKANN_SERVICE_PROFILE_DIR": str(profile)},
                        )
                    else:
                        is_core = arm == "navhints"
                        result = run_one(
                            args.nav_bin, label, held, args.heldout_gt, args.index_prefix,
                            L, args.out, mode="nav", ivf=args.ivf,
                            extra_env={
                                "DISKANN_SERVICE_PROFILE_DIR": str(profile),
                                "DISKANN_SERVICE_MODE": arm,
                                "DISKANN_EXPERIENCE_CACHE_CAPACITY": CORE["recent"] if is_core else 0,
                                "DISKANN_EXPERIENCE_HUB_CAPACITY": 0,
                                "DISKANN_LOCAL_CACHE_MODE": "fifo",
                                "DISKANN_LOCAL_RECENT_STARTS": CORE["starts"] if is_core else 0,
                            },
                        )
                    summary = profile_summary(profile / f"L{L}.csv")
                    if abs(result["reads"] - summary["mean_reads"]) > 1.0:
                        # A one-read discrepancy may arise from arithmetic/counter conventions,
                        # but larger disagreement is evidence of the wrong suffix or profiler.
                        raise RuntimeError(f"{label}: per-query reads mismatch native result")
                    measurements[label] = {"recall": result["recall"], **summary}
                    save(args.out / "measurements.partial.json", measurements)
    finally:
        os.sched_setaffinity(0, set(cpus))

    matches = matched_recall(measurements, widths, args.reps, seeds)
    qualified = [row for row in matches if row["available"]]
    report = {
        "scope": "PubMed1M, one in-flight request; no multi-client throughput inference",
        "engine": "same pinned SSD DiskANN, author-aligned Catapult routing NOT original Catapult engine",
        "causality": "fixed 5K static prefix, 4K causal warmup and 1K measured; feedback after each completed query",
        "controller": CORE,
        "widths": widths, "repetitions": args.reps, "catapult_seeds": seeds,
        "metric_definitions": {
            "service_us": "whole per-query wall time from immediately before search through post-query feedback",
            "engine_us": "engine/adapter timer, includes routing/PQ/SSD search; excludes post-query feedback",
            "remainder_us": "service_us - engine_us: result marshaling, instrumentation and feedback; not feedback only",
            "io_us": "native provider aggregate I/O time, may overlap CPU; not additive to engine time",
            "serial_throughput_qps": "measured queries / sum(service_us), concurrency one, not concurrent QPS",
        },
        "data_limits": "identical frozen query sequence in every rotation; percentile spreads are not confidence intervals",
        "diagnostic": args.reps < 3 or len(seeds) < 3 or len(widths) < 8,
        "catapult_snapshots": snapshots, "measurements": measurements,
        "matched_recall": matches, "qualified_common_targets": len(qualified),
    }
    if qualified:
        report["mean_navhints_latency_saved_vs_catapult_percent"] = statistics.mean(
            row["navhints_latency_saved_vs_catapult_percent"] for row in qualified
        )
    save(args.out / "complete-service-pubmed.json", report)
    print(f"COMPLETE_SERVICE_CAUSAL_PASS points={len(qualified)} diagnostic={report['diagnostic']}",
          flush=True)


if __name__ == "__main__":
    main()
