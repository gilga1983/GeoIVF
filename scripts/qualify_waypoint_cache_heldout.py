#!/usr/bin/env python3
"""Held-out evaluation of learned portal-region endpoint/waypoint caches at k=1."""
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
METHODS = (
    "portal",
    "uniform-endpoint-lru10",
    "global-endpoint-s1",
    "global-endpoint-s2",
    "global-endpoint-s4",
    "waypoint-s1",
    "waypoint-s2",
    "waypoint-s4",
)


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
    if not (0 <= start < rows):
        raise ValueError("invalid suffix start")
    nrows = rows - start
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)
    return nrows, dim


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


def run_one(binary, work, out, tag, queries, gt, index_prefix, router, cache_file):
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
        "num_nodes_to_cache": None,
        "search_io_limit": None,
        "post_processor": None,
    }
    cfg = {
        "search_directories": [str(work)],
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
    env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
    env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    for name in (
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)
    if cache_file is not None:
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(cache_file)

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

    rr = result_rows(json.loads(output.read_text()))
    if len(rr) != 1 or int(rr[0]["search_l"]) != K:
        raise ValueError(f"{tag}: unexpected result rows")
    row = dict(rr[0])
    if float(row["recall"]) != -1.0:
        raise ValueError("recall sentinel missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--portal-router", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for attr in ("binary", "queries", "gt", "index_prefix", "portal_router", "cache_dir", "work", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != 10000 or dim != 768 or args.train_rows != 5000:
        raise ValueError(f"expected frozen 10000x768 split 5000/5000, got {rows}x{dim}")

    heldout = args.work / "heldout-5000.fbin"
    nheld, _ = suffix_fbin(args.queries, heldout, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    cache_map = {
        "portal": None,
        "uniform-endpoint-lru10": args.cache_dir / "uniform-endpoint-lru10.bin",
        "global-endpoint-s1": args.cache_dir / "global-endpoint-s1.bin",
        "global-endpoint-s2": args.cache_dir / "global-endpoint-s2.bin",
        "global-endpoint-s4": args.cache_dir / "global-endpoint-s4.bin",
        "waypoint-s1": args.cache_dir / "waypoint-s1.bin",
        "waypoint-s2": args.cache_dir / "waypoint-s2.bin",
        "waypoint-s4": args.cache_dir / "waypoint-s4.bin",
    }
    for name, p in cache_map.items():
        if p is not None and not p.is_file():
            raise FileNotFoundError(f"{name}: {p}")

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
                # Rotate order each repetition to reduce residual first/last bias.
                shift = rep % len(METHODS)
                order = METHODS[shift:] + METHODS[:shift]
                for method in order:
                    tag = f"r{rep}-{method}"
                    print(f"RUN {tag}", flush=True)
                    row = run_one(
                        args.binary,
                        args.work,
                        args.out,
                        tag,
                        heldout,
                        args.gt,
                        args.index_prefix,
                        args.portal_router,
                        cache_map[method],
                    )
                    all_rows.append({"method": method, "rep": rep, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)
    summary = {}
    base_rows = [r for r in all_rows if r["method"] == "portal"]
    base = {
        "qps": avg(base_rows, "qps"),
        "mean_latency_us": avg(base_rows, "mean_latency"),
        "mean_ios": avg(base_rows, "mean_ios"),
        "mean_io_us": avg(base_rows, "mean_io_time"),
        "mean_cpu_us": avg(base_rows, "mean_cpu_time"),
        "mean_hops": avg(base_rows, "mean_hops"),
        "mean_comparisons": avg(base_rows, "mean_comparisons"),
    }
    summary["portal"] = base

    for method in METHODS[1:]:
        rr = [r for r in all_rows if r["method"] == method]
        obj = {
            "qps": avg(rr, "qps"),
            "mean_latency_us": avg(rr, "mean_latency"),
            "mean_ios": avg(rr, "mean_ios"),
            "mean_io_us": avg(rr, "mean_io_time"),
            "mean_cpu_us": avg(rr, "mean_cpu_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }
        obj["qps_speedup_vs_portal"] = obj["qps"] / base["qps"]
        obj["io_reduction_fraction_vs_portal"] = 1.0 - obj["mean_ios"] / base["mean_ios"]
        obj["latency_reduction_fraction_vs_portal"] = (
            1.0 - obj["mean_latency_us"] / base["mean_latency_us"]
        )
        summary[method] = obj

    result = {
        "workload": "reconstructed MedRAG-Zipf heldout suffix",
        "split": {"train_prefix": 5000, "heldout_suffix": 5000},
        "search": {
            "k": K,
            "search_l": K,
            "beam_width": IO_BEAM,
            "threads": THREADS,
            "repetitions": args.reps,
            "recall": "skipped in first I/O-generalization experiment",
        },
        "summary": summary,
    }
    save(args.out / "waypoint-heldout-k1-result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
