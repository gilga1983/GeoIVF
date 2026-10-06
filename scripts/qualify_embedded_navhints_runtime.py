#!/usr/bin/env python3
"""Evaluate continuation NavHints embedded in DiskANN associated data."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from qualify_regional_navhints_runtime import (
    K,
    LS,
    THREADS,
    aggregate,
    fbin_shape,
    interp,
    payload_ivf,
    result_rows,
    save,
    suffix_fbin,
    suffix_gt,
)
import subprocess
import time


RUN_TIMEOUT_SECONDS = 600


def run_one(binary, out, tag, queries, gt, index_prefix, start_ivf, embedded_limit):
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
                    "search_list": list(LS),
                    "beam_width": 8,
                    "recall_at": K,
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
    logp = out / f"{tag}.log"
    save(inp, cfg)

    env = os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_EMBEDDED_HINTS",
        "DISKANN_REGIONAL_HINT_FILE",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_PROGRESSIVE_HINT_TOPK",
        "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)

    env["DISKANN_HINT_IVF_FILE"] = str(start_ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    env["DISKANN_EMBEDDED_HINTS"] = str(embedded_limit)

    cmd = [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)]
    started = time.monotonic()
    with logp.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise TimeoutError(tag)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    rows = sorted(result_rows(json.loads(output.read_text())), key=lambda r: int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS):
        raise ValueError(f"{tag}: bad L grid")
    print(f"completed {tag} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "gt5000", "index-prefix", "ivf-16k", "embed-manifest", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    for name in ("binary", "queries", "gt5000", "index_prefix", "ivf_16k", "embed_manifest", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected workload")
    embedded_meta = json.loads(args.embed_manifest.read_text())
    if embedded_meta["physical_file_size_change"] != 0:
        raise ValueError("embedded index is not physical-I/O neutral")

    held = args.work / "heldout-last1000.fbin"
    gt = args.work / "heldout-last1000.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

    methods = ("canonical16k", "embedded4", "embedded8")
    limits = {"canonical16k": 0, "embedded4": 4, "embedded8": 8}
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
                        limits[method],
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    row = rr[2]
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
    base = summary["canonical16k"]

    same_l = {}
    for l in LS:
        b = base[str(l)]
        item = {}
        for method in methods:
            row = summary[method][str(l)]
            item[method] = {
                "recall_percent": row["recall_percent"],
                "mean_ios": row["mean_ios"],
                "latency_us": row["median_latency_us"],
                "cpu_us": row["median_cpu_us"],
                "mean_comparisons": row["mean_comparisons"],
                "recall_delta_points_vs_canonical": row["recall_percent"] - b["recall_percent"],
                "io_change_percent_vs_canonical": 100.0 * (row["mean_ios"] / b["mean_ios"] - 1.0),
                "latency_change_percent_vs_canonical": 100.0 * (
                    row["median_latency_us"] / b["median_latency_us"] - 1.0
                ),
                "cpu_change_percent_vs_canonical": 100.0 * (
                    row["median_cpu_us"] / b["median_cpu_us"] - 1.0
                ),
                "comparison_delta_vs_canonical": row["mean_comparisons"] - b["mean_comparisons"],
            }
        same_l[str(l)] = item

    fixed = {}
    lo = max(summary[m]["10"]["recall_percent"] for m in methods)
    hi = min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0, 45.0, 55.0, 65.0):
        if not lo <= target <= hi:
            continue
        canonical = {
            "mean_ios": interp(base, target, "mean_ios"),
            "latency_us": interp(base, target, "median_latency_us"),
            "cpu_us": interp(base, target, "median_cpu_us"),
            "mean_comparisons": interp(base, target, "mean_comparisons"),
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
                cur["mean_ios"] / canonical["mean_ios"] - 1.0
            )
            cur["latency_change_percent_vs_canonical"] = 100.0 * (
                cur["latency_us"] / canonical["latency_us"] - 1.0
            )
            cur["cpu_change_percent_vs_canonical"] = 100.0 * (
                cur["cpu_us"] / canonical["cpu_us"] - 1.0
            )
            arms[method] = cur
        fixed[str(target)] = arms

    ranking = []
    for method in ("embedded4", "embedded8"):
        ranking.append({
            "method": method,
            "mean_fixed_recall_latency_change_percent": sum(
                fixed[t][method]["latency_change_percent_vs_canonical"] for t in fixed
            ) / len(fixed),
            "mean_fixed_recall_io_change_percent": sum(
                fixed[t][method]["io_change_percent_vs_canonical"] for t in fixed
            ) / len(fixed),
        })
    ranking.sort(key=lambda x: x["mean_fixed_recall_latency_change_percent"])

    result = {
        "workload": "MedRAG-Zipf final 1000 queries; exact PubMed1M IP top-16 truth",
        "training": "first 9000 queries, canonical 16K NavHints L64 traces; 256-region residual maps embedded into every node",
        "storage": embedded_meta,
        "policy": {
            "bootstrap": "canonical optimized 16K Hint-IVF",
            "embedded_payload": "8 u32 continuation IDs per graph node in DiskANN associated data",
            "runtime_arms": {"canonical16k": 0, "embedded4": 4, "embedded8": 8},
            "per_beam": "consult closest expanded node only; skip consecutive duplicate payload; admit at most one best unvisited hint through fixed-L gate",
        },
        "memory_payload_bytes": {
            "bootstrap_hint_ivf": payload_ivf(args.ivf_16k),
            "incremental_resident_region_table": 0,
            "embedded_bytes_per_node": embedded_meta["associated_data_bytes"],
        },
        "evaluation": {
            "K": K,
            "Ls": list(LS),
            "beam": 8,
            "threads": THREADS,
            "repetitions": args.reps,
        },
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
        "ranking_by_mean_fixed_recall_latency": ranking,
    }
    save(args.out / "embedded-navhints-runtime.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
