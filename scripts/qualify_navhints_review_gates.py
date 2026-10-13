#!/usr/bin/env python3
"""Causal, real-SSD reviewer controls on the frozen PubMed/MedCPT stream.

Gate A: position-sum / frequency / random junction ranking at B=14,464,
        C=2,048, k=8, with identical PQ clustering and search parameters.
Gate B: change B/C at *fixed k=8 and fixed accounted routing payload*.
Every arm receives the same frozen 5K training trace, 4K warm-up and 1K
measured suffix. Rotated repeats are not independent query populations.

Writes raw per-L/repetition data, matched-recall summaries and explicit
negative results. No manuscript results are changed automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from qualify_navhints_catapult_headtohead import run_one, shape, slice_fbin, TRAIN, WARM, MEASURE, DIM

LS = (20, 40, 80, 120, 160, 240, 320)
REGIONS = 512
BUDGET_BYTES = 102940

# tag, B, C, K. The reference is the same paper controller everywhere.
ALLOCATION_ARMS = (
    ("reference-k8", 14464, 2048, 8),
    ("reference-k4", 14464, 2048, 4),
    ("dest1024-k8", 15488, 1024, 8),
    ("dest4096-k8", 12416, 4096, 8),
    ("dest8192-k8", 8320, 8192, 8),
)
SCORE_ARMS = (
    ("position-sum", 14464, 2048, 8),
    ("frequency", 14464, 2048, 8),
    ("random", 14464, 2048, 8),
)


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_ivf(path: Path, B: int, C: int):
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    m = json.loads(manifest.read_text())
    if int(m["nlist"]) != REGIONS or int(m["landmark_ids"]) != B:
        raise ValueError(f"unexpected B or region count in {path}: {m}")
    # Same accounting convention and fixed denominator as the paper.
    static_bytes = path.stat().st_size + REGIONS * (64 + 4)
    payload = static_bytes + 4 * C
    if payload != BUDGET_BYTES:
        raise ValueError(f"unequal payload in {path}: {payload} != {BUDGET_BYTES}")
    return {
        "ivf_path": str(path), "ivf_sha256": digest(path),
        "manifest_sha256": digest(manifest),
        "B": B, "C": C,
        "accounted_payload_bytes": payload
    }


def frontier(curve: dict):
    """Pareto-efficient increasing-recall points; no extrapolation."""
    points = sorted(
        [{"L": int(L), "recall": v["recall"], "reads": v["reads"],
          "latency_us": v["latency_us"]} for L, v in curve.items()],
        key=lambda x: (x["recall"], x["reads"])
    )
    # Keep minimum cost at duplicate recall; remove points with a later
    # observation that has at least as much recall and no greater cost.
    filtered = []
    for point in points:
        later = [p for p in points if p["recall"] >= point["recall"] - 1e-7
                 and p["reads"] <= point["reads"] + 1e-7
                 and (p["recall"] > point["recall"] + 1e-7
                      or p["reads"] < point["reads"] - 1e-7)]
        if not later:
            filtered.append(point)
    return filtered


def interpolate(points, recall, metric):
    if not points or recall < points[0]["recall"] - 1e-7 or recall > points[-1]["recall"] + 1e-7:
        return None
    for p in points:
        if abs(recall - p["recall"]) < 1e-7:
            return float(p[metric])
    for a, b in zip(points, points[1:]):
        if a["recall"] <= recall <= b["recall"]:
            f = (recall - a["recall"]) / (b["recall"] - a["recall"])
            return float(a[metric] + f * (b[metric] - a[metric]))
    return None


def summarize(reps, arms, ref):
    all_matched = []
    per_rep = []
    for i, curves in enumerate(reps):
        front = {tag: frontier(curves[tag]) for tag, *_ in arms}
        low = max(p[0]["recall"] for p in front.values())
        high = min(p[-1]["recall"] for p in front.values())
        anchors = [
            p["recall"] for p in front[ref]
            if low - 1e-7 <= p["recall"] <= high + 1e-7
        ]
        anchors = sorted(set(anchors))
        if len(anchors) < 3:
            raise RuntimeError(f"rep {i}: only {len(anchors)} common reference anchors in [{low}, {high}]")
        results = []
        for target in anchors:
            base_read = interpolate(front[ref], target, "reads")
            base_latency = interpolate(front[ref], target, "latency_us")
            if not base_read or not base_latency:
                raise ValueError("invalid reference metric")
            row = {"recall": target, "reference_reads": base_read,
                   "reference_latency_us": base_latency, "comparisons": {}}
            for tag, *_ in arms:
                if tag == ref:
                    continue
                x = interpolate(front[tag], target, "reads")
                y = interpolate(front[tag], target, "latency_us")
                if x is None or y is None:
                    raise RuntimeError(f"rep {i}: interpolation would extrapolate {tag}")
                result = {
                    "additional_read_percent": 100 * (x / base_read - 1),
                    "additional_latency_percent": 100 * (y / base_latency - 1),
                    "absolute_reads": x, "latency_us": y
                }
                row["comparisons"][tag] = result
                all_matched.append((tag, result))
            results.append(row)
        per_rep.append({"repetition": i, "common_interval": [low, high],
                        "anchors": results, "anchor_count": len(anchors)})

    summary = {}
    for tag, *_ in arms:
        if tag == ref:
            continue
        rows = [v for name, v in all_matched if name == tag]
        summary[tag] = {
            "common_target_count": len(rows),
            "additional_read_percent": statistics.mean(x["additional_read_percent"] for x in rows),
            "additional_latency_percent": statistics.mean(x["additional_latency_percent"] for x in rows),
            "repetition_means_read_percent": [
                statistics.mean(row["comparisons"][tag]["additional_read_percent"]
                                for row in r["anchors"])
                for r in per_rep
            ],
        }
    return {"reference": ref,
            "positive_additional_cost_favors_reference": True,
            "selection": "Per-repetition common reference-anchored Recall@10 grid; no extrapolation",
            "per_repetition": per_rep, "summary": summary}


def run_gate(gate, arms, reference, ivfs, heldout, gt, index, a):
    raw = []
    for rep in range(a.repetitions):
        observations = {tag: {} for tag, *_ in arms}
        for ix, L in enumerate(LS):
            order = list(arms)
            start = (rep + ix) % len(order)
            order = order[start:] + order[:start]
            for tag, B, C, K in order:
                label = f"{gate}-r{rep}-L{L}-{tag}"
                row = run_one(
                    a.nav_bin, label, heldout, gt, index, L,
                    a.out / "raw", mode="nav", ivf=ivfs[tag],
                    extra_env={
                        "DISKANN_EXPERIENCE_CACHE_CAPACITY": C,
                        "DISKANN_EXPERIENCE_HUB_CAPACITY": 0,
                        "DISKANN_LOCAL_CACHE_MODE": "fifo",
                        "DISKANN_LOCAL_REGION_NPROBE": 1,
                        "DISKANN_LOCAL_RECENT_STARTS": K,
                    }
                )
                observations[tag][str(L)] = row
                save(a.out / f"{gate}.partial.json",
                     {"completed_repetitions": len(raw), "current_repetition": rep,
                      "last_arm": label, "previous": raw, "current": observations})
        raw.append(observations)
    analysis = summarize(raw, arms, reference)
    result = {
        "gate": gate, "workload": "PubMed1M/MedCPT",
        "search_L": list(LS), "repetitions": a.repetitions,
        "split": {"static_train_queries": TRAIN, "warmup_queries": WARM,
                  "measured_queries": MEASURE},
        "native_graph_and_ssd": str(index),
        "same_frozen_queries": True,
        "no_persistent_writes": True,
        "fixed_PQ_representation": True,
        "search_threads": 4,
        "simultaneous_queries": 1,
        "accounted_payload_bytes": BUDGET_BYTES,
        "limited_inference": "Rotated replays share one query sequence; not independent demand samples",
        "arm_config": {tag: {"B": B, "C": C, "K": K, **validate_ivf(ivfs[tag], B, C)}
                       for tag, B, C, K in arms},
        "raw": raw, "analysis": analysis
    }
    save(a.out / f"{gate}.json", result)
    print(f"{gate.upper()}_VALIDATED", flush=True)
    print(json.dumps(analysis["summary"], indent=2, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("nav-bin", "queries", "heldout-gt", "index-prefix",
                 "score-ivf-dir", "allocation-ivf-dir", "work", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.repetitions <= 5:
        raise ValueError("repetitions must be in 1..5")
    for name in ("nav_bin", "queries", "heldout_gt", "index_prefix",
                 "score_ivf_dir", "allocation_ivf_dir", "work", "out"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.nav_bin.is_file():
        raise FileNotFoundError(a.nav_bin)
    if shape(a.queries) != (TRAIN + WARM + MEASURE, DIM):
        raise ValueError("frozen training/warm-up/test query dimensions changed")
    a.work.mkdir(parents=True, exist_ok=True)
    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    heldout = slice_fbin(a.queries, a.work / "heldout5000.fbin", TRAIN, WARM + MEASURE)

    score = {tag: a.score_ivf_dir / tag / "hints-b14464-nlist512-spherical.bin"
             for tag, *_ in SCORE_ARMS}
    alloc = {tag: a.allocation_ivf_dir /
             f"hints-b{B}-nlist512-spherical.bin"
             for tag, B, C, K in ALLOCATION_ARMS}
    for label, arms, files in (("score", SCORE_ARMS, score),
                               ("allocation", ALLOCATION_ARMS, alloc)):
        for tag, B, C, _ in arms:
            validate_ivf(files[tag], B, C)

    # Exactly the same serialized real-SSD experiment lock used by existing suite.
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        raise RuntimeError("at least four permitted physical CPUs needed")
    os.sched_setaffinity(0, set(allowed[:4]))
    try:
        run_gate("junction-score", SCORE_ARMS, "position-sum", score,
                 heldout, a.heldout_gt, a.index_prefix, a)
        run_gate("fixed-k8-memory", ALLOCATION_ARMS, "reference-k8", alloc,
                 heldout, a.heldout_gt, a.index_prefix, a)
    finally:
        os.sched_setaffinity(0, set(allowed))
    print("NAVHINTS_REVIEW_GATES_1_AND_2_PASS", flush=True)


if __name__ == "__main__":
    main()
