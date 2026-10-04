#!/usr/bin/env python3
"""Four-arm SSD evaluation on reconstructed MedRAG-Zipf.

Arms:
  1. medoid
  2. paper CatapultDB
  3. static IP portal
  4. static IP portal + paper CatapultDB

All arms use the same pinned DiskANN disk index, query order, search budget, I/O
beam, thread count, and no node cache. Throughput runs intentionally skip exact
recall; recall validation is a separate sampled experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_K = (1, 2, 4, 8, 16)
THREADS = 4
IO_BEAM = 8
CATAPULT_SEEDS = (0, 1, 2)
PINNED_DISKANN = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size mismatch: {path.stat().st_size} != {expected}")
    return rows, dim


def prefix_fbin(src: Path, dst: Path, nrows: int) -> Path:
    rows, dim = fbin_shape(src)
    if nrows <= 0 or nrows >= rows:
        return src
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<II", nrows, dim))
        remaining = nrows * dim * 4
        while remaining:
            block = fin.read(min(8 << 20, remaining))
            if not block:
                raise ValueError("unexpected EOF while slicing query fbin")
            fout.write(block)
            remaining -= len(block)
    return dst


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj and "recall" in obj:
            found.append(obj)
        else:
            for v in obj.values():
                found.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(result_rows(v))
    return found


def source_load(prefix: Path):
    return {
        "disk-index-source": "Load",
        "data_type": "float32",
        "load_path": str(prefix),
    }


def run_disk(
    binary: Path,
    work: Path,
    out: Path,
    tag: str,
    prefix: Path,
    queries: Path,
    gt: Path,
    k: int,
    method: str,
    seed: int,
):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [k],
        "beam_width": IO_BEAM,
        "recall_at": k,
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
        "jobs": [
            {
                "type": "disk-index",
                "content": {
                    "source": source_load(prefix),
                    "search_phase": phase,
                },
            }
        ],
    }
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    for name in (
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
    ):
        env.pop(name, None)

    use_catapult = method in ("catapult", "portal-catapult")
    use_portal = method in ("portal", "portal-catapult")
    if use_catapult:
        env["DISKANN_PAPER_CATAPULT"] = "1"
        env["DISKANN_CATAPULT_HASHES"] = "8"
        env["DISKANN_CATAPULT_CAPACITY"] = "40"
        env["DISKANN_CATAPULT_SEED"] = str(seed)
    if use_portal:
        router = os.environ.get("MEDRAG_ZIPF_PORTAL_ROUTER")
        if not router:
            raise RuntimeError("MEDRAG_ZIPF_PORTAL_ROUTER is required for portal arms")
        env["DISKANN_IP_PORTAL_ROUTER_FILE"] = router
        env["DISKANN_IP_PORTAL_NPROBE"] = "32"

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
    if int(row["search_l"]) != k:
        raise ValueError(f"{tag}: wrong k/L row")
    if float(row["recall"]) != -1.0:
        raise ValueError(f"{tag}: recall skip sentinel missing")
    return row


def avg(rs, key):
    return float(np.mean([float(r[key]) for r in rs]))


def summarize(rows, k_values):
    result = {}
    for k in k_values:
        result[str(k)] = {}
        for method in ("medoid", "catapult", "portal", "portal-catapult"):
            rs = [r for r in rows if r["k"] == k and r["method"] == method]
            if not rs:
                raise ValueError(f"missing {method} rows for k={k}")
            obj = {
                "rounds": len(rs),
                "qps": avg(rs, "qps"),
                "mean_latency_us": avg(rs, "mean_latency"),
                "p95_latency_us": avg(rs, "p95_latency"),
                "mean_ios": avg(rs, "mean_ios"),
                "mean_io_us": avg(rs, "mean_io_time"),
                "mean_cpu_us": avg(rs, "mean_cpu_time"),
                "mean_comparisons": avg(rs, "mean_comparisons"),
                "mean_hops": avg(rs, "mean_hops"),
                "catapult_usage_percent": avg(rs, "catapult_usage_percentage"),
                "mean_catapult_starts": avg(rs, "mean_catapult_starts"),
            }
            result[str(k)][method] = obj

        base = result[str(k)]["medoid"]
        for method in ("catapult", "portal", "portal-catapult"):
            m = result[str(k)][method]
            m["qps_speedup_vs_medoid"] = m["qps"] / base["qps"]
            m["latency_reduction_fraction_vs_medoid"] = (
                1.0 - m["mean_latency_us"] / base["mean_latency_us"]
            )
            m["io_reduction_fraction_vs_medoid"] = 1.0 - m["mean_ios"] / base["mean_ios"]
            m["hop_reduction_fraction_vs_medoid"] = 1.0 - m["mean_hops"] / base["mean_hops"]

        cat = result[str(k)]["catapult"]
        comb = result[str(k)]["portal-catapult"]
        portal = result[str(k)]["portal"]
        comb["qps_speedup_vs_catapult"] = comb["qps"] / cat["qps"]
        comb["io_reduction_fraction_vs_catapult"] = 1.0 - comb["mean_ios"] / cat["mean_ios"]
        comb["qps_speedup_vs_portal"] = comb["qps"] / portal["qps"]
        comb["io_reduction_fraction_vs_portal"] = 1.0 - comb["mean_ios"] / portal["mean_ios"]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--portal-router", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k-values", default="1,2,4,8,16")
    args = ap.parse_args()

    for attr in ("binary", "queries", "gt", "index_prefix", "portal_router", "work", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reps < 1 or args.reps > len(CATAPULT_SEEDS):
        raise ValueError("reps must be in 1..3")
    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("invalid k-values")

    rows_total, dim = fbin_shape(args.queries)
    if dim != 768:
        raise ValueError(f"expected 768-D MedCPT queries, got {dim}")
    qfile = prefix_fbin(args.queries, args.work / "queries-prefix.fbin", args.max_queries)
    used_rows, _ = fbin_shape(qfile)

    if not args.portal_router.is_file():
        raise FileNotFoundError(args.portal_router)
    os.environ["MEDRAG_ZIPF_PORTAL_ROUTER"] = str(args.portal_router)

    # Keep the whole benchmark on the same four CPUs.
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("runner exposes fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Serialize SSD performance measurements with the rest of the GeoIVF
        # qualification harness. Without this lock, sibling self-hosted runners
        # can inject millisecond-scale device-latency spikes into one arm only.
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)
            for k in k_values:
                for rep in range(args.reps):
                    seed = CATAPULT_SEEDS[rep]
                    # Interleave arms to reduce temporal host drift.
                    for method in ("medoid", "portal", "catapult", "portal-catapult"):
                        tag = f"k{k}-{method}-r{rep}-seed{seed}"
                        print(f"RUN {tag}", flush=True)
                        row = run_disk(
                            args.binary,
                            args.work,
                            args.out,
                            tag,
                            args.index_prefix,
                            qfile,
                            args.gt,
                            k,
                            method,
                            seed,
                        )
                        rows.append(
                            {
                                "k": k,
                                "method": method,
                                "rep": rep,
                                "seed": seed if "catapult" in method else -1,
                                **row,
                            }
                        )
                        save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = summarize(rows, k_values)
    result = {
        "workload": "reconstructed MedRAG-Zipf",
        "queries_total_available": rows_total,
        "queries_used": used_rows,
        "query_dim": dim,
        "diskann_revision": PINNED_DISKANN,
        "index": {
            "dataset": "first 1,000,000 PubMed MedCPT article embeddings",
            "metric": "inner_product",
            "node_cache": None,
        },
        "paper_catapult": {
            "hashes": 8,
            "capacity": 40,
            "seeds": list(CATAPULT_SEEDS[:args.reps]),
        },
        "portal": {
            "nlist": 1024,
            "nprobe": 32,
            "metric": "inner_product",
            "router": str(args.portal_router),
        },
        "search": {
            "k_values": list(k_values),
            "query_threads": THREADS,
            "ssd_io_beam_width": IO_BEAM,
            "repetitions": args.reps,
            "recall": "skipped in throughput run",
        },
        "summary": summary,
    }
    save(args.out / "medrag-zipf-four-arm-result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
