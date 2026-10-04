#!/usr/bin/env python3
"""Same-workload comparison of low-RAM DiskANN navigation baselines.

Frozen protocol:
* first 5000 MedRAG-Zipf queries are training/history only;
* final 5000 queries are held out;
* same PubMed1M DiskANN graph, PQ codes, SSD layout, L=1, beam=8, four threads;
* three order-rotated repetitions under the shared SSD timing lock.

Arms:
* medoid: ordinary DiskANN;
* cache-900: native DiskANN BFS full-node cache, near the 3 MiB budget;
* cache-1024: deliberately generous native cache whose vector+edge payload alone
  already exceeds the learned-navigation budget;
* qsev: DiskANN++-style query-sensitive entry pool, 1024 FP32 entries.
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
IO_BEAM = 8
K = 1
CACHE_COUNTS = (900, 1024)
MAX_DEGREE = 64
TARGET_BYTES = 3_159_860


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def suffix_fbin(src: Path, dst: Path, start: int):
    rows, dim = fbin_shape(src)
    n = rows - start
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", n, dim))
        remaining = n * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)
    return n, dim


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            found.append(obj)
        else:
            for v in obj.values():
                found.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(result_rows(v))
    return found


def run_one(binary, out, tag, queries, gt, index_prefix, *, cache_nodes=None, qsev=None):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [K],
        "beam_width": IO_BEAM,
        "recall_at": K,
        "num_threads": THREADS,
        "is_flat_search": False,
        "distance": "inner_product",
        "vector_filters_file": None,
        "num_nodes_to_cache": cache_nodes,
        "search_io_limit": None,
        "post_processor": None,
    }
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
                "search_phase": phase,
            },
        }],
    }
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    for name in (
        "DISKANN_QSEV_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)
    if qsev is not None:
        env["DISKANN_QSEV_FILE"] = str(qsev)

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
        raise ValueError(f"{tag}: expected one result row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) != -1.0:
        raise ValueError(f"{tag}: throughput-only recall sentinel missing")
    return row


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def median(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--qsev", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "qsev", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != 10000 or dim != 768 or args.train_rows != 5000:
        raise ValueError("expected frozen 10000x768 workload with 5000/5000 split")
    held = args.work / "heldout.fbin"
    nheld, _ = suffix_fbin(args.queries, held, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    qsev_bytes = args.qsev.stat().st_size
    if qsev_bytes > TARGET_BYTES:
        raise ValueError(f"QSEV pool exceeds target bytes: {qsev_bytes} > {TARGET_BYTES}")

    methods = ["medoid", "cache-900", "cache-1024", "qsev"]
    all_rows = []

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
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
                for method in order:
                    if method == "medoid":
                        row = run_one(
                            args.binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix,
                        )
                    elif method == "qsev":
                        row = run_one(
                            args.binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix, qsev=args.qsev,
                        )
                    else:
                        count = int(method.split("-", 1)[1])
                        row = run_one(
                            args.binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix, cache_nodes=count,
                        )
                    all_rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)
    summary = {}
    for method in methods:
        rr = [r for r in all_rows if r["method"] == method]
        summary[method] = {
            "rounds": len(rr),
            "mean_ios": mean(rr, "mean_ios"),
            "median_qps": median(rr, "qps"),
            "median_latency_us": median(rr, "mean_latency"),
            "median_io_time_us": median(rr, "mean_io_time"),
            "median_cpu_time_us": median(rr, "mean_cpu_time"),
            "mean_hops": mean(rr, "mean_hops"),
            "mean_comparisons": mean(rr, "mean_comparisons"),
            "mean_cache_hit_percent": mean(rr, "cache_hit_percentage"),
        }

    cache_memory = {}
    for n in CACHE_COUNTS:
        vectors = n * dim * 4
        max_edges = n * MAX_DEGREE * 4
        payload = vectors + max_edges
        cache_memory[str(n)] = {
            "nodes": n,
            "vector_payload_bytes": vectors,
            "max_degree_edge_payload_bytes": max_edges,
            "vector_plus_max_degree_edge_payload_bytes": payload,
            "vector_plus_edge_payload_mib": payload / (1 << 20),
            "note": (
                "excludes HashMap storage, Vec headers/capacity slack, allocator "
                "metadata, and other Rust object overhead"
            ),
        }

    result = {
        "workload": "reconstructed MedRAG-Zipf heldout suffix",
        "split": {"train_prefix": 5000, "heldout_suffix": 5000},
        "search": {
            "k": K,
            "beam": IO_BEAM,
            "threads": THREADS,
            "metric": "inner_product",
            "same_diskann_graph_and_pq": True,
        },
        "target_extra_ram_bytes": TARGET_BYTES,
        "target_extra_ram_mib": TARGET_BYTES / (1 << 20),
        "qsev_bytes": qsev_bytes,
        "qsev_mib": qsev_bytes / (1 << 20),
        "qsev_budget_fraction": qsev_bytes / TARGET_BYTES,
        "native_cache_memory_accounting": cache_memory,
        "summary": summary,
        "interpretation_guardrails": [
            "cache-1024 is intentionally generous: vector+max-degree edge payload already exceeds target before allocator/hash overhead",
            "QSEV exact centroid representatives strengthen the published approximate centroid-to-vertex construction",
            "QSEV routing time is inside both end-to-end QPS and per-query latency",
            "throughput campaign skips recall; exact-recall validation should follow any competitive result",
        ],
    }
    save(args.out / "navigation-memory-baselines.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
