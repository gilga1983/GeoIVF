#!/usr/bin/env python3
"""Small real-SSD multi-client latency stress test for the final NavHints Core.

Load P=1,2,4 simultaneous *independent* causal clients sharing one SSD.
Each client has its own FIFO state and runs one query at a time; thus this
is OS/device contention, NOT a shared-history concurrent ANN engine.
We compare 4 and 8 recent starts at fixed B=14464, C=2048 to quantify
the incremental history cost under load. CPU and SSD lock are external.

Clients reuse the same 5K query population, with separate deterministic
rotations of warmup and measured portions (paired between k=4 and k=8).
Report measured response time, I/O, quality and per-client throughput.
DO NOT infer production tail latency or aggregate QPS from wall time.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import struct
from pathlib import Path

import numpy as np

from qualify_navhints_catapult_headtohead import (
    run_one, slice_fbin, shape, TRAIN, WARM, MEASURE, DIM
)

LS = (80, 160)
LEVELS = (1, 2, 4)
STARTS = (4, 8)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def rotate_replay(src, gt, client, out):
    """Shift warm & measured order together with GT, without mixing splits."""
    out.mkdir(parents=True, exist_ok=True)
    qdata = np.memmap(src, mode="r", dtype="<f4", offset=8, shape=(WARM + MEASURE, DIM))
    iw = np.roll(np.arange(WARM), client * 997)
    it = WARM + np.roll(np.arange(MEASURE), client * 251)
    perm = np.concatenate([iw, it])
    query = out / f"client{client}.fbin"
    with query.open("wb") as f:
        f.write(struct.pack("<II", WARM + MEASURE, DIM))
        np.asarray(qdata[perm], dtype="<f4").tofile(f)

    rows, k = struct.unpack("<II", gt.read_bytes()[:8])
    if rows != WARM + MEASURE or k < 10:
        raise ValueError("ground truth must align with 5K held-out replay")
    bytes_per_field = rows * k * 4
    payload = gt.read_bytes()
    has_distances = len(payload) == 8 + 2 * bytes_per_field
    if len(payload) not in (8 + bytes_per_field, 8 + 2 * bytes_per_field):
        raise ValueError("invalid ground truth byte count")
    ids = np.frombuffer(payload, dtype="<u4", count=rows * k, offset=8).reshape(rows, k)
    target = out / f"client{client}.gt"
    with target.open("wb") as f:
        f.write(struct.pack("<II", rows, k))
        np.asarray(ids[perm], dtype="<u4").tofile(f)
        if has_distances:
            distances = np.frombuffer(
                payload, dtype="<f4", count=rows * k,
                offset=8 + bytes_per_field
            ).reshape(rows, k)
            np.asarray(distances[perm], dtype="<f4").tofile(f)
    return query, target


def describe(rows):
    return {
        "clients": len(rows),
        "median_client_mean_latency_us": statistics.median(r["latency_us"] for r in rows),
        "mean_client_mean_latency_us": statistics.mean(r["latency_us"] for r in rows),
        "mean_client_reads": statistics.mean(r["reads"] for r in rows),
        "mean_client_recall": statistics.mean(r["recall"] for r in rows),
        "sum_native_client_qps": sum(r["qps"] for r in rows),
        "note": "sum_native_client_qps is the sum of overlapping native per-client estimates, not wall-clock measurement of a stable service",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("nav-bin", "queries", "heldout-gt", "index-prefix",
                 "ivf", "work", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=2)
    a = p.parse_args()
    if a.repetitions not in (1, 2, 3):
        raise ValueError("repetitions must be 1, 2 or 3")
    for name in ("nav_bin", "queries", "heldout_gt",
                 "index_prefix", "ivf", "work", "out"):
        setattr(a, name, getattr(a, name).resolve())
    if shape(a.queries) != (TRAIN + WARM + MEASURE, DIM):
        raise ValueError("frozen query shape mismatch")
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    heldout = slice_fbin(a.queries, a.work / "heldout5000.fbin",
                         TRAIN, WARM + MEASURE)
    clients = [rotate_replay(heldout, a.heldout_gt, i, a.work / "clients")
               for i in range(max(LEVELS))]

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 16:
        raise RuntimeError("need 16 permitted CPUs for 4 independent four-thread clients")
    # Cap the stress load to 16 logical CPUs; no hidden multi-node jobs.
    os.sched_setaffinity(0, set(allowed[:16]))
    result = {
        "study": "multi-client contention / incremental destination-start CPU and SSD cost",
        "scope": "independent causal per-process FIFO histories, common frozen SSD graph and query population",
        "no_shared_online_history": True,
        "same_clients_nesting_across_load": True,
        "rotations_preserve_warmup_and_measured_slices": True,
        "load_processes": list(LEVELS), "starts": list(STARTS),
        "search_L": list(LS), "repetitions": a.repetitions,
        "individual_client_search_threads": 4,
        "total_permitted_cpus": 16,
        "not_comparable_to_single_engine_multithreading": True,
        "not_independent_query_populations": True,
        "raw": [], "aggregates": {},
    }
    try:
        for rep in range(a.repetitions):
            for L in LS:
                arms = list(STARTS if rep % 2 == 0 else reversed(STARTS))
                for k in arms:
                    loads = list(LEVELS if (rep + k // 4) % 2 == 0 else reversed(LEVELS))
                    for P in loads:
                        label = f"r{rep}-L{L}-k{k}-clients{P}"
                        print("CONCURRENCY", label, flush=True)
                        def one(i):
                            query, gt = clients[i]
                            return run_one(
                                a.nav_bin, f"{label}-client{i}", query, gt,
                                a.index_prefix, L, a.out, mode="nav", ivf=a.ivf,
                                extra_env={
                                    "DISKANN_EXPERIENCE_CACHE_CAPACITY": 2048,
                                    "DISKANN_EXPERIENCE_HUB_CAPACITY": 0,
                                    "DISKANN_LOCAL_CACHE_MODE": "fifo",
                                    "DISKANN_LOCAL_REGION_NPROBE": 1,
                                    "DISKANN_LOCAL_RECENT_STARTS": k,
                                })
                        with concurrent.futures.ThreadPoolExecutor(max_workers=P) as pool:
                            rows = list(pool.map(one, range(P)))
                        result["raw"].append({
                            "repetition": rep, "L": L, "starts": k, "client_count": P,
                            "per_client": rows, "summary": describe(rows)
                        })
                        save(a.out / "multi-client.partial.json", result)
    finally:
        os.sched_setaffinity(0, set(allowed))

    for L in LS:
        for k in STARTS:
            for P in LEVELS:
                observations = [v["summary"] for v in result["raw"]
                                if v["L"] == L and v["starts"] == k and v["client_count"] == P]
                result["aggregates"][f"L{L}-k{k}-P{P}"] = {
                    "median_latency_us": statistics.median(v["median_client_mean_latency_us"]
                                                           for v in observations),
                    "mean_reads": statistics.mean(v["mean_client_reads"] for v in observations),
                    "mean_recall": statistics.mean(v["mean_client_recall"] for v in observations),
                    "sum_native_client_qps": statistics.median(v["sum_native_client_qps"]
                                                                for v in observations)
                }
    save(a.out / "multi-client-latency.json", result)
    print("NAVHINTS_MULTI_CLIENT_STRESS_PASS", flush=True)
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
