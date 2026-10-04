#!/usr/bin/env python3
"""Run the CatapultDB paper's Uniform workload in the SSD DiskANN harness.

The index is built once and cached. Measurements compare unmodified medoid
DiskANN with the paper Catapult policy at k={1,2,4,8,16}, 4 query threads,
and fixed SSD I/O beam concurrency. Recall is explicitly skipped for the
150k throughput stream; exact recall validation is a separate experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

K_VALUES = (1, 2, 4, 8, 16)
CATAPULT_SEEDS = (0, 1, 2)
THREADS = 4
IO_BEAM = 8
DIM = 768
PINNED_DISKANN = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


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


def one_query(src: Path, dst: Path) -> None:
    with src.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
        if rows < 1 or dim != DIM:
            raise ValueError("bad uniform query file")
        vector = f.read(DIM * 4)
        if len(vector) != DIM * 4:
            raise ValueError("truncated query file")
    with dst.open("wb") as f:
        f.write(struct.pack("<II", 1, DIM))
        f.write(vector)


def source_build(base: Path, prefix: Path):
    return {
        "disk-index-source": "Build",
        "data_type": "float32",
        "data": str(base),
        "distance": "inner_product",
        "dim": DIM,
        "max_degree": 64,
        "l_build": 100,
        "num_threads": 4,
        "build_ram_limit_gb": 12.0,
        "num_pq_chunks": 64,
        "quantization_type": "FP",
        "save_path": str(prefix),
    }


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
    source: dict,
    queries: Path,
    gt: Path,
    k: int,
    *,
    catapult_seed: int | None,
    threads: int = THREADS,
):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [k],
        "beam_width": IO_BEAM,
        "recall_at": k,
        "num_threads": threads,
        "is_flat_search": False,
        "distance": "inner_product",
        "vector_filters_file": None,
        "num_nodes_to_cache": None,
        "search_io_limit": None,
        "post_processor": None,
    }
    cfg = {
        "search_directories": [str(work)],
        "jobs": [{"type": "disk-index", "content": {"source": source, "search_phase": phase}}],
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
    ):
        env.pop(name, None)
    if catapult_seed is not None:
        env["DISKANN_PAPER_CATAPULT"] = "1"
        env["DISKANN_CATAPULT_HASHES"] = "8"
        env["DISKANN_CATAPULT_CAPACITY"] = "40"
        env["DISKANN_CATAPULT_SEED"] = str(catapult_seed)

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1 or int(rows[0]["search_l"]) != k:
        raise ValueError(f"{tag}: unexpected result rows")
    row = dict(rows[0])
    if float(row["recall"]) != -1.0:
        raise ValueError(f"{tag}: throughput-only recall sentinel missing")
    return row


def index_ready(prefix: Path) -> bool:
    # A successful Rust DiskANN disk index has these three files.
    candidates = list(prefix.parent.glob(prefix.name + "*"))
    names = [p.name for p in candidates if p.is_file()]
    return (
        any("pq_pivots" in n for n in names)
        and any("pq_compressed" in n for n in names)
        and any("disk.index" in n or n.endswith("_disk.index") for n in names)
    )


def summarize(rows):
    result = {}
    for k in K_VALUES:
        medoid = [r for r in rows if r["k"] == k and r["method"] == "medoid"]
        cat = [r for r in rows if r["k"] == k and r["method"] == "catapult"]
        if len(medoid) != 3 or len(cat) != 3:
            raise ValueError(f"missing rows for k={k}")

        def avg(key, rs):
            return float(np.mean([float(r[key]) for r in rs]))

        m = {
            "rounds": len(medoid),
            "qps": avg("qps", medoid),
            "mean_latency_us": avg("mean_latency", medoid),
            "mean_ios": avg("mean_ios", medoid),
            "mean_io_us": avg("mean_io_time", medoid),
            "mean_cpu_us": avg("mean_cpu_time", medoid),
            "mean_comparisons": avg("mean_comparisons", medoid),
            "mean_hops": avg("mean_hops", medoid),
        }
        c = {
            "seeds": [int(r["seed"]) for r in cat],
            "qps": avg("qps", cat),
            "mean_latency_us": avg("mean_latency", cat),
            "mean_ios": avg("mean_ios", cat),
            "mean_io_us": avg("mean_io_time", cat),
            "mean_cpu_us": avg("mean_cpu_time", cat),
            "mean_comparisons": avg("mean_comparisons", cat),
            "mean_hops": avg("mean_hops", cat),
            "catapult_usage_percent": avg("catapult_usage_percentage", cat),
            "mean_catapult_starts": avg("mean_catapult_starts", cat),
        }
        c["qps_speedup_vs_medoid"] = c["qps"] / m["qps"]
        c["latency_reduction_fraction_vs_medoid"] = 1.0 - c["mean_latency_us"] / m["mean_latency_us"]
        c["io_reduction_fraction_vs_medoid"] = 1.0 - c["mean_ios"] / m["mean_ios"]
        c["hop_reduction_fraction_vs_medoid"] = 1.0 - c["mean_hops"] / m["mean_hops"]
        c["comparison_reduction_fraction_vs_medoid"] = (
            1.0 - c["mean_comparisons"] / m["mean_comparisons"]
        )
        result[str(k)] = {"medoid": m, "catapult": c}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-cache", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.binary = args.binary.resolve()
    args.base = args.base.resolve()
    args.queries = args.queries.resolve()
    args.gt = args.gt.resolve()
    args.index_cache = args.index_cache.resolve()
    args.work = args.work.resolve()
    args.out = args.out.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    args.index_cache.mkdir(parents=True, exist_ok=True)

    prefix = args.index_cache / "diskann-index"
    ready = args.index_cache / "READY-v1"
    build_query = args.work / "build-one-query.fbin"
    one_query(args.queries, build_query)

    lock_path = args.index_cache.parent / (args.index_cache.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ready.exists():
            for p in args.index_cache.iterdir():
                if p.name != ready.name:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
            print("Building shared 1M PubMed SSD index", flush=True)
            row = run_disk(
                args.binary,
                args.work,
                args.out,
                "index-build-smoke",
                source_build(args.base, prefix),
                build_query,
                args.gt,
                1,
                catapult_seed=None,
                threads=1,
            )
            if not index_ready(prefix):
                raise RuntimeError("DiskANN build completed but expected index files are missing")
            ready.write_text(
                json.dumps(
                    {
                        "diskann_revision": PINNED_DISKANN,
                        "base": str(args.base),
                        "dim": DIM,
                        "metric": "inner_product",
                        "max_degree": 64,
                        "l_build": 100,
                        "num_pq_chunks": 64,
                        "quantization_type": "FP",
                        "build_smoke_row": row,
                    },
                    indent=2,
                )
                + "\n"
            )

    rows = []
    allowed = sorted(os.sched_getaffinity(0))
    # Let DiskANN itself use four query threads; constrain the process to four CPUs.
    cpu_set = set(allowed[: min(THREADS, len(allowed))])
    if len(cpu_set) < THREADS:
        raise RuntimeError("runner exposes fewer than four CPUs")
    os.sched_setaffinity(0, cpu_set)
    try:
        for k in K_VALUES:
            # Interleave baseline repetitions and the three paper randomness seeds.
            for rep, seed in enumerate(CATAPULT_SEEDS):
                m = run_disk(
                    args.binary,
                    args.work,
                    args.out,
                    f"k{k}-medoid-r{rep}",
                    source_load(prefix),
                    args.queries,
                    args.gt,
                    k,
                    catapult_seed=None,
                )
                rows.append({"k": k, "method": "medoid", "round": rep, "seed": -1, **m})

                c = run_disk(
                    args.binary,
                    args.work,
                    args.out,
                    f"k{k}-catapult-seed{seed}",
                    source_load(prefix),
                    args.queries,
                    args.gt,
                    k,
                    catapult_seed=seed,
                )
                rows.append({"k": k, "method": "catapult", "round": rep, "seed": seed, **c})
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = summarize(rows)
    save(
        args.out / "paper-uniform-ssd-result.json",
        {
            "workload": "CatapultDB paper Uniform",
            "base": "first 1,000,000 PubMed MedCPT article embeddings",
            "queries": "150,000 iid uniform [-1,1]^768, seed 0",
            "diskann_revision": PINNED_DISKANN,
            "paper_catapult": {"hashes": 8, "capacity": 40, "seeds": list(CATAPULT_SEEDS)},
            "search": {
                "k_values": list(K_VALUES),
                "query_threads": THREADS,
                "ssd_io_beam_width": IO_BEAM,
                "metric": "inner_product",
                "node_cache": None,
                "recall": "explicitly skipped for throughput stream",
            },
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
