#!/usr/bin/env python3
"""Held-out ablation over geometric region counts.

Each region count has its own independently built portal router, traversal trace,
and fixed-budget skip-weighted heavy-hitter cache learned from the first 5000
queries. Evaluation uses the same final 5000 queries.

To isolate the value of partitioning from "more starts", every cache arm exposes
at most 40 learned IDs per query.
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
MAX_CACHE_STARTS = 40


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


def cache_header(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != b"GIWPT001":
        raise ValueError(f"bad cache {path}")
    nlist, total = struct.unpack("<II", raw[8:16])
    return int(nlist), int(total)


def run_one(binary, out, tag, queries, gt, index_prefix, router, cache):
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
                    "search_list": [K],
                    "beam_width": IO_BEAM,
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
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
    env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    for name in (
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)
    if cache is not None:
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(cache)
        env["DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"] = str(MAX_CACHE_STARTS)

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
    return dict(rr[0])


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def median(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--router-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--regions", default="1,64,128,256,512,1024,2048")
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for a in ("binary","queries","gt","index_prefix","router_root","cache_root","work","out"):
        setattr(args, a, getattr(args, a).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != 10000 or dim != 768 or args.train_rows != 5000:
        raise ValueError("expected frozen 10000x768, split 5000/5000")
    held = args.work / "heldout.fbin"
    nheld, _ = suffix_fbin(args.queries, held, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    regions = [int(x) for x in args.regions.split(",") if x.strip()]
    paths = {}
    cache_meta = {}
    for nlist in regions:
        router = args.router_root / f"nlist-{nlist}" / "router.bin"
        cache = args.cache_root / f"nlist-{nlist}" / "heavy-skip-b2500.bin"
        if not router.is_file() or not cache.is_file():
            raise FileNotFoundError(f"nlist={nlist}: {router} / {cache}")
        cn, total = cache_header(cache)
        if cn != nlist:
            raise ValueError(f"cache/router nlist mismatch for {nlist}")
        paths[nlist] = (router, cache)
        cache_meta[str(nlist)] = {"cache_ids": total, "cache_bytes": cache.stat().st_size}

    methods = []
    for n in regions:
        methods += [f"portal-{n}", f"cache-{n}"]

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
                    kind, raw_n = method.split("-", 1)
                    n = int(raw_n)
                    router, cache = paths[n]
                    row = run_one(
                        args.binary, args.out, f"r{rep}-{method}", held, args.gt,
                        args.index_prefix, router, cache if kind == "cache" else None,
                    )
                    all_rows.append({"rep": rep, "method": method, "regions": n, "kind": kind, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)
    summary = {}
    for n in regions:
        for kind in ("portal", "cache"):
            rr = [r for r in all_rows if r["regions"] == n and r["kind"] == kind]
            summary[f"{kind}-{n}"] = {
                "regions": n,
                "kind": kind,
                "mean_ios": mean(rr, "mean_ios"),
                "median_qps": median(rr, "qps"),
                "median_latency_us": median(rr, "mean_latency"),
                "median_io_time_us": median(rr, "mean_io_time"),
                "median_cpu_time_us": median(rr, "mean_cpu_time"),
                "mean_hops": mean(rr, "mean_hops"),
            }

    curve = []
    for n in regions:
        p = summary[f"portal-{n}"]
        c = summary[f"cache-{n}"]
        curve.append({
            "regions": n,
            "portal_mean_ios": p["mean_ios"],
            "cache_mean_ios": c["mean_ios"],
            "cache_median_qps": c["median_qps"],
            "cache_io_reduction_vs_own_portal": 1.0 - c["mean_ios"] / p["mean_ios"],
            **cache_meta[str(n)],
        })

    result = {
        "workload": "reconstructed MedRAG-Zipf heldout suffix",
        "split": {"train_prefix": 5000, "heldout_suffix": 5000},
        "region_counts": regions,
        "cache_policy": "global skip-weighted navigation heavy hitters",
        "cache_budget_requested": 2500,
        "max_cached_starts_per_query": MAX_CACHE_STARTS,
        "nprobe_requested": 32,
        "summary": summary,
        "curve": curve,
    }
    save(args.out / "region-count-ablation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
