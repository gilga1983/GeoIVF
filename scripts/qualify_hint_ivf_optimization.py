#!/usr/bin/env python3
"""A/B the reference and optimized Hint-IVF implementations."""
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
L = 1
NPROBE = 8


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def suffix_fbin(src: Path, dst: Path, start: int):
    rows, dim = fbin_shape(src)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remaining = (rows - start) * dim * 4
        while remaining:
            block = fin.read(min(16 << 20, remaining))
            if not block:
                raise ValueError("truncated fbin")
            fout.write(block)
            remaining -= len(block)


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            found.append(obj)
        else:
            for value in obj.values():
                found.extend(result_rows(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(result_rows(value))
    return found


def run_one(binary, out, tag, queries, gt, index_prefix, ivf):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [L],
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
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = str(NPROBE)
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1:
        raise ValueError(f"{tag}: expected one result row, got {len(rows)}")
    row = dict(rows[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-binary", type=Path, required=True)
    ap.add_argument("--optimized-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--ivf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    for name in (
        "reference_binary", "optimized_binary", "queries", "gt",
        "index_prefix", "ivf", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    methods = ("reference", "optimized")
    binaries = {
        "reference": args.reference_binary,
        "optimized": args.optimized_binary,
    }

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                order = methods if rep % 2 == 0 else tuple(reversed(methods))
                for method in order:
                    row = run_one(
                        binaries[method],
                        args.out,
                        f"r{rep}-{method}",
                        held,
                        args.gt,
                        args.index_prefix,
                        args.ivf,
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        summary[method] = {
            "rounds": len(rr),
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "median_pq_preprocess_us": med(rr, "mean_pq_preprocess_time"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
            "mean_hops": avg(rr, "mean_hops"),
        }

    ref = summary["reference"]
    opt = summary["optimized"]
    # Routing is intended to select the same candidate set. Fail closed if
    # search quality or graph work shifts materially.
    if abs(opt["recall_percent"] - ref["recall_percent"]) > 1e-6:
        raise RuntimeError(
            f"optimized recall changed: {ref['recall_percent']} -> {opt['recall_percent']}"
        )
    if abs(opt["mean_ios"] - ref["mean_ios"]) > 1e-9:
        raise RuntimeError(
            f"optimized I/O changed: {ref['mean_ios']} -> {opt['mean_ios']}"
        )
    if abs(opt["mean_hops"] - ref["mean_hops"]) > 1e-9:
        raise RuntimeError(
            f"optimized hops changed: {ref['mean_hops']} -> {opt['mean_hops']}"
        )

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "configuration": {
            "hints": 16000,
            "nlist": 512,
            "nprobe": NPROBE,
            "K": K,
            "L": L,
            "beam": IO_BEAM,
            "threads": THREADS,
        },
        "summary": summary,
        "optimized_over_reference": {
            "qps_ratio": opt["median_qps"] / ref["median_qps"],
            "latency_ratio": opt["median_latency_us"] / ref["median_latency_us"],
            "cpu_ratio": opt["median_cpu_us"] / ref["median_cpu_us"],
            "pq_preprocess_ratio": (
                opt["median_pq_preprocess_us"] / ref["median_pq_preprocess_us"]
                if ref["median_pq_preprocess_us"] else None
            ),
            "comparison_ratio": opt["mean_comparisons"] / ref["mean_comparisons"],
        },
    }
    save(args.out / "hint-ivf-optimization.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
