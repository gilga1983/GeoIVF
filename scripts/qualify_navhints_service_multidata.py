#!/usr/bin/env python3
"""Complete-service SSD measurements for the three non-PubMed NavHints workloads.

This *extends* the successful pinned PubMed experiment.  It does not change
NavHints, Catapult, the SSD indexes, the recall definition, or recorded
competitor results.  Its own new competitor measurements are necessary because
latency must be obtained in a single controlled device epoch.

One in-flight request preserves post-query causal feedback.  All routing arms
run the SAME warm-up and held-out query stream per workload.  Catapult trains
on the disjoint 5K prefix before replay; no online state leaks across runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
from pathlib import Path

from qualify_navhints_service_latency import (
    CORE, K, BEAM, SEARCH_THREADS, profile_summary, percentile,
    interpolate, save,
)
from qualify_navhints_catapult_headtohead import rows_of, snapshot_entries

STATIC_TRAIN = 5000
HASH_BITS = 8
DEFAULT_WIDTHS = "10,20,40,80,160,320"
WORKLOADS = {
    "coveo": {"warmup": 20000, "measure": 5000, "data_type": "float32",
              "distance": "inner_product", "dim": 50},
    "text2image-10M": {"warmup": 4000, "measure": 1000, "data_type": "float32",
                       "distance": "inner_product", "dim": 200},
    "bigann-10M": {"warmup": 4000, "measure": 1000, "data_type": "uint8",
                   "distance": "squared_l2", "dim": 128},
}
NATIVE_ARMS = ("baseline", "junctions", "navhints")


def shape(path: Path, itemsize: int) -> tuple[int, int]:
    with path.open("rb") as file:
        raw = file.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated vector header: {path}")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * itemsize:
        raise ValueError(f"invalid vector length: {path}")
    return rows, dim


def truth_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        raw = file.read(8)
    if len(raw) != 8:
        raise ValueError("truncated ground-truth header")
    rows, k = struct.unpack("<II", raw)
    expect = 8 + rows * k * 4
    if path.stat().st_size not in (expect, expect + rows * k * 4):
        raise ValueError("unexpected ground-truth layout")
    return rows, k


def make_unscored_training_truth(path: Path) -> None:
    """The disjoint 5K training phase does NOT evaluate recall or use oracle IDs.

    The pinned benchmark requires a syntactically valid GT file even when
    DISKANN_SKIP_RECALL=1, so provide zero-filled synthetic IDs explicitly.
    """
    path.write_bytes(struct.pack("<II", STATIC_TRAIN, K) +
                     b"\0" * (STATIC_TRAIN * K * 4))


def catapult_capacity(payload: int, dim: int) -> tuple[int, int]:
    """Do not give Catapult a smaller *accounted* routing budget than Core."""
    hash_bytes = HASH_BITS * dim * 4
    if payload <= hash_bytes:
        raise ValueError("NavHints payload smaller than Catapult's hyperplanes")
    each_bucket = (1 << HASH_BITS) * 4
    capacity = max(80, math.ceil((payload - hash_bytes) / each_bucket))
    return capacity, hash_bytes + capacity * each_bucket


def one_run(binary: Path, label: str, queries: Path, gt: Path,
            prefix: Path, L: int, out: Path, cfg: dict, *,
            mode: str, warmup: int, capacity: int, ivf: Path | None = None,
            seed: int | None = None, snapshot_load: Path | None = None,
            snapshot_dump: Path | None = None, train: bool = False) -> dict:
    data_type, distance = cfg["data_type"], cfg["distance"]
    phase = {
        "queries": str(queries), "groundtruth": str(gt), "search_list": [L],
        "beam_width": BEAM, "recall_at": K, "num_threads": SEARCH_THREADS,
        "is_flat_search": False, "distance": distance, "vector_filters_file": None,
        "num_nodes_to_cache": None, "search_io_limit": None, "post_processor": None,
    }
    job = {"search_directories": [str(out)],
           "jobs": [{"type": "disk-index", "content": {
               "source": {"disk-index-source": "Load", "data_type": data_type,
                          "load_path": str(prefix)}, "search_phase": phase}}]}
    inp, result, log = (
        out / f"{label}.input.json", out / f"{label}.output.json",
        out / f"{label}.log"
    )
    save(inp, job)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISKANN_")}
    if mode == "catapult":
        if seed is None:
            raise ValueError("Catapult seed missing")
        env.update({
            "DISKANN_PAPER_CATAPULT": "1",
            "DISKANN_CATAPULT_HASHES": str(HASH_BITS),
            "DISKANN_CATAPULT_CAPACITY": str(capacity),
            "DISKANN_CATAPULT_SEED": str(seed),
            "DISKANN_CATAPULT_CAUSAL_REPLAY": "1",
            "DISKANN_CATAPULT_CAUSAL_WARMUP": str(0 if train else warmup),
        })
        if snapshot_load is not None:
            env["DISKANN_CATAPULT_SNAPSHOT_LOAD"] = str(snapshot_load)
        if snapshot_dump is not None:
            env["DISKANN_CATAPULT_SNAPSHOT_DUMP"] = str(snapshot_dump)
        if train:
            env["DISKANN_SKIP_RECALL"] = "1"
    elif mode in NATIVE_ARMS:
        if train or ivf is None:
            raise ValueError("NavHints routing arms require a frozen directory")
        env.update({
            "DISKANN_HINT_IVF_FILE": str(ivf),
            "DISKANN_HINT_IVF_NPROBE": str(CORE["nprobe"]),
            "DISKANN_HINT_IVF_MAX_STARTS": "1",
            "DISKANN_EXPERIENCE_REPLAY": "1",
            "DISKANN_EXPERIENCE_WARMUP": str(warmup),
            "DISKANN_EXPERIENCE_HUB_CAPACITY": "0",
            "DISKANN_EXPERIENCE_CACHE_CAPACITY": str(CORE["recent"] if mode == "navhints" else 0),
            "DISKANN_LOCAL_CACHE_MODE": "fifo",
            "DISKANN_LOCAL_RECENT_STARTS": str(CORE["starts"] if mode == "navhints" else 0),
            "DISKANN_SERVICE_MODE": mode,
        })
    else:
        raise ValueError(f"invalid engine arm {mode}")
    profile = out / "profiles" / label
    if not train:
        env["DISKANN_SERVICE_PROFILE_DIR"] = str(profile)

    with log.open("w") as file:
        subprocess.run([str(binary), "run", "--input-file", str(inp),
                        "--output-file", str(result)], env=env,
                       stdout=file, stderr=subprocess.STDOUT,
                       timeout=7200, check=True)
    rows = list(rows_of(json.loads(result.read_text())))
    if len(rows) != 1 or int(rows[0]["search_l"]) != L:
        raise RuntimeError(f"{label}: malformed native result")
    row = rows[0]
    if train:
        if float(row["recall"]) != -1.0:
            raise RuntimeError("Catapult offline training scored oracle recall")
        if snapshot_dump is None or not snapshot_dump.is_file():
            raise RuntimeError("missing Catapult training snapshot")
        return {"training": True}
    if not 0 <= float(row["recall"]) <= 100:
        raise RuntimeError(f"{label}: invalid recall")
    if mode == "catapult":
        check = f"CATAPULT_CAUSAL_STATS L={L} warmup={warmup} measured={cfg['measure']}"
        if check not in log.read_text():
            raise RuntimeError(f"{label}: missing strict Catapult causal evidence")
    elif mode == "navhints":
        if "EXPERIENCE_STATS" not in log.read_text() or "eval_writes=0" not in log.read_text():
            raise RuntimeError(f"{label}: Core persistence unexpectedly active")
    summary = profile_summary(profile / f"L{L}.csv", warmup=warmup, measured=cfg["measure"])
    if abs(float(row["mean_ios"]) - summary["mean_reads"]) > 1:
        raise RuntimeError(f"{label}: native reads mismatch profiled reads")
    return {"recall": float(row["recall"]), **summary}


def interpolated_across_repetitions(measurements: dict, widths: list[int], reps: int,
                                    seeds: list[int]) -> tuple[list[dict], list[dict]]:
    def curves(rep: int, arm: str) -> dict:
        return {L: measurements[f"r{rep}-L{L}-{arm}"] for L in widths}

    triple_rows, four_rows = [], []
    for L in widths:
        target = statistics.mean(curves(rep, "baseline")[L]["recall"]
                                 for rep in range(reps))
        entry = {"baseline_L": L, "recall_at_10": target}
        variants = {a: [curves(rep, a) for rep in range(reps)]
                    for a in NATIVE_ARMS}
        variants["catapult"] = [curves(rep, f"catapult-s{seed}")
                                 for rep in range(reps) for seed in seeds]
        matched = {}
        for arm, reps_curves in variants.items():
            pts = [interpolate(c, target) for c in reps_curves]
            if any(p is None for p in pts):
                matched[arm] = None
                continue
            matched[arm] = {
                key: statistics.mean(p[key] for p in pts)
                for key in ("mean_service_us", "mean_engine_us", "mean_remainder_us",
                            "mean_io_us", "mean_reads")
            }
            matched[arm]["per_run_service_us"] = [p["mean_service_us"] for p in pts]
            matched[arm]["median_of_run_service_us"] = statistics.median(
                p["mean_service_us"] for p in pts
            )
        entry["arms"] = matched
        if all(matched[a] is not None for a in NATIVE_ARMS):
            entry["savings_percent"] = {
                comparator: 100 * (1 - matched["navhints"]["mean_service_us"] /
                                   matched[comparator]["mean_service_us"])
                for comparator in ("baseline", "junctions")
            }
            triple_rows.append(entry)
            if matched["catapult"] is not None:
                entry["savings_percent"]["catapult"] = 100 * (
                    1 - matched["navhints"]["mean_service_us"] /
                    matched["catapult"]["mean_service_us"])
                four_rows.append(entry)
    return triple_rows, four_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=tuple(WORKLOADS), required=True)
    for name in ("nav-bin", "cat-bin", "train-queries", "replay", "replay-gt",
                 "index-prefix", "ivf", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--widths", default=DEFAULT_WIDTHS)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cat-seeds", default="0")
    args = ap.parse_args()
    cfg = WORKLOADS[args.dataset]
    widths = sorted({int(x) for x in args.widths.split(",")})
    seeds = sorted({int(x) for x in args.cat_seeds.split(",")})
    if not widths or any(L < K for L in widths):
        raise ValueError("invalid search width grid")
    if not seeds or any(seed not in (0, 1, 2) for seed in seeds):
        raise ValueError("invalid Catapult seeds")
    if args.reps < 1 or args.reps > 3:
        raise ValueError("invalid repetition count")
    for attr in ("nav_bin", "cat_bin", "train_queries", "replay", "replay_gt", "ivf"):
        p = getattr(args, attr).resolve()
        setattr(args, attr, p)
        if not p.is_file():
            raise FileNotFoundError(p)
    args.index_prefix = args.index_prefix.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    itemsize = {"float32": 4, "uint8": 1}[cfg["data_type"]]
    if shape(args.train_queries, itemsize) != (STATIC_TRAIN, cfg["dim"]):
        raise RuntimeError("not the frozen disjoint 5K training queries")
    if shape(args.replay, itemsize) != (cfg["warmup"] + cfg["measure"], cfg["dim"]):
        raise RuntimeError("not the frozen causal replay")
    gt_count, gt_k = truth_shape(args.replay_gt)
    if gt_count != cfg["warmup"] + cfg["measure"] or gt_k < K:
        raise RuntimeError("ground truth does not match replay")
    ivf_manifest = json.loads(args.ivf.with_suffix(args.ivf.suffix + ".manifest.json").read_text())
    B = int(ivf_manifest["landmark_ids"])
    if not 0 < B <= CORE["junctions"] or int(ivf_manifest["nlist"]) != CORE["nlist"]:
        raise RuntimeError("not the selected trained junction directory")
    if args.dataset != "coveo" and B != CORE["junctions"]:
        raise RuntimeError("public benchmark must use the full junction allocation")
    payload = args.ivf.stat().st_size + CORE["nlist"] * (64 + 4) + CORE["recent"] * 4
    capacity, cat_bytes = catapult_capacity(payload, cfg["dim"])
    args.work.joinpath("training-sentinel.gt").parent.mkdir(parents=True, exist_ok=True)
    train_gt = args.work / "training-sentinel.gt"
    make_unscored_training_truth(train_gt)
    profiles = {}
    snapshots = {}
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < SEARCH_THREADS:
        raise RuntimeError("fewer than four available CPU cores")
    os.sched_setaffinity(0, set(cpus[:SEARCH_THREADS]))
    try:
        for L in widths:
            for seed in seeds:
                label = f"cat-train-L{L}-s{seed}"
                snapshot = args.work / f"{label}.snapshot"
                one_run(args.cat_bin, label, args.train_queries, train_gt,
                        args.index_prefix, L, args.out, cfg, mode="catapult",
                        warmup=cfg["warmup"], capacity=capacity, seed=seed,
                        snapshot_dump=snapshot, train=True)
                occupied = snapshot_entries(snapshot)
                if not 0 < occupied <= (1 << HASH_BITS) * capacity:
                    raise RuntimeError("invalid Catapult snapshot size")
                snapshots[label] = {
                    "resident_entries": occupied,
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                }
                save(args.out / "snapshots.partial.json", snapshots)
        for rep in range(args.reps):
            order_widths = widths[rep % len(widths):] + widths[:rep % len(widths)]
            for j, L in enumerate(order_widths):
                arms = list(NATIVE_ARMS) + [f"catapult-s{seed}" for seed in seeds]
                shift = (j + rep) % len(arms)
                arms = arms[shift:] + arms[:shift]
                print(f"SERVICE dataset={args.dataset} rep={rep} L={L} "
                      f"order={','.join(arms)}", flush=True)
                for arm in arms:
                    label = f"r{rep}-L{L}-{arm}"
                    if arm.startswith("catapult-s"):
                        seed = int(arm.rsplit("s", 1)[1])
                        result = one_run(
                            args.cat_bin, label, args.replay, args.replay_gt,
                            args.index_prefix, L, args.out, cfg, mode="catapult",
                            warmup=cfg["warmup"], capacity=capacity, seed=seed,
                            snapshot_load=args.work / f"cat-train-L{L}-s{seed}.snapshot",
                        )
                    else:
                        result = one_run(
                            args.nav_bin, label, args.replay, args.replay_gt,
                            args.index_prefix, L, args.out, cfg, mode=arm,
                            warmup=cfg["warmup"], capacity=capacity, ivf=args.ivf,
                        )
                    profiles[label] = result
                    save(args.out / "measurements.partial.json", profiles)
                    print(f"PASS {label}: recall={result['recall']:.2f} "
                          f"service={result['mean_service_us']:.2f}us "
                          f"p95={result['service_p95_us']:.1f}us "
                          f"reads={result['mean_reads']:.1f}", flush=True)
    finally:
        os.sched_setaffinity(0, set(cpus))

    three, four = interpolated_across_repetitions(profiles, widths, args.reps, seeds)
    summary = {}
    for comparator in ("baseline", "junctions", "catapult"):
        rows = four if comparator == "catapult" else three
        vals = [r["savings_percent"][comparator] for r in rows]
        summary[comparator] = {
            "common_targets": len(vals),
            "mean_service_latency_saving_percent": statistics.mean(vals) if vals else None,
            "min_saving_percent": min(vals) if vals else None,
            "max_saving_percent": max(vals) if vals else None,
            "negative_targets": sum(v < 0 for v in vals),
        }
    report = {
        "dataset": args.dataset,
        "source": "frozen original disk-resident DiskANN SSD index, no graph modifications",
        "data_type": cfg["data_type"], "distance": cfg["distance"],
        "split": {"static_training_queries": STATIC_TRAIN,
                  "causal_warmup_queries": cfg["warmup"],
                  "measured_queries": cfg["measure"]},
        "core": {**CORE, "actual_junctions": B, "accounted_payload_bytes": payload},
        "catapult": {"hash_bits": HASH_BITS, "capacity_per_bucket": capacity,
                     "accounted_payload_bytes": cat_bytes, "seeds": seeds,
                     "source_aligned_policy_port": True},
        "protocol": {"recall_at": K, "beam": BEAM, "threads": SEARCH_THREADS,
                     "in_flight_queries": 1, "widths": widths,
                     "rotated_repetitions": args.reps,
                     "timing_scope": "routing, search, copy results, online history update",
                     "latency_us": "measured complete-request wall clock",
                     "tail_latency": "directly measured per-L per-arm p50/p95/p99, not interpolated",
                     "mean_recall_matching": "interpolated ABSOLUTE means only, no extrapolation"},
        "limitations": "Fixed query suffix, same device; repeat runs do not imply independent query-population confidence intervals; serial QPS only; Catapult is author-aligned routing on DiskANN not native CatapultDB; memory is accounted payload not process RSS.",
        "diagnostic": args.reps < 3 or len(seeds) < 3 or len(widths) < 6,
        "training_snapshots": snapshots,
        "measurements": profiles,
        "matched_three_arm": three,
        "matched_four_arm": four,
        "summary": summary,
    }
    save(args.out / f"complete-service-{args.dataset}.json", report)
    print(f"COMPLETE_SERVICE_MULTIDATA_PASS dataset={args.dataset} "
          f"common_three={len(three)} common_four={len(four)} "
          f"diagnostic={report['diagnostic']}", flush=True)


if __name__ == "__main__":
    main()
