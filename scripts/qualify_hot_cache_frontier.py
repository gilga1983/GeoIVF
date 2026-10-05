#!/usr/bin/env python3
"""Compare workload-hot full-node DiskANN caches against NavHints.

The hot-cache arms preload IDs learned from the same 5K training traversal
history. Caching changes only whether expanded graph records require SSD I/O;
it does not alter entry selection or search L, so recall should match vanilla
DiskANN exactly at each L.

NavHints uses the canonical packed-direct 16K/512/p8 deployment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

import numpy as np

THREADS = 4
BEAM = 8
K = 10
LS = (10, 20, 40, 80, 160, 320)
HOT_COUNTS = (33, 128, 512, 1024)
DIM = 768
FLOAT_BYTES = 4


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            out.append(obj)
        else:
            for v in obj.values():
                out.extend(rows(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(rows(v))
    return out


def run(binary, out, tag, queries, gt, index_prefix, *, hot_ids=None, ivf=None):
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
                    "beam_width": BEAM,
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
    save(inp, cfg)
    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_STATIC_CACHE_IDS_FILE",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)
    if hot_ids is not None:
        env["DISKANN_STATIC_CACHE_IDS_FILE"] = str(hot_ids)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = "8"
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
        )
    rr = sorted(rows(json.loads(output.read_text())), key=lambda x: int(x["search_l"]))
    if [int(x["search_l"]) for x in rr] != list(LS):
        raise ValueError(f"{tag}: bad L rows")
    return rr


def aggregate(reps):
    by_l = {}
    for rr in reps:
        for r in rr:
            by_l.setdefault(int(r["search_l"]), []).append(r)
    result = {}
    for l in LS:
        x = by_l[l]
        result[str(l)] = {
            "recall_percent": float(np.mean([float(r["recall"]) for r in x])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in x])),
            "median_qps": float(np.median([float(r["qps"]) for r in x])),
            "median_latency_us": float(np.median([float(r["mean_latency"]) for r in x])),
            "mean_cache_hit_percent": float(
                np.mean([float(r["cache_hit_percentage"]) for r in x])
            ),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--hot-dir", type=Path, required=True)
    ap.add_argument("--ivf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","hot_dir","ivf","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.out.mkdir(parents=True, exist_ok=True)

    methods = ["baseline", "navhints"] + [f"hot-{n}" for n in HOT_COUNTS]
    hot_files = {}
    for n in HOT_COUNTS:
        p = args.hot_dir / f"hot-cache-n{n}.bin"
        if not p.is_file():
            raise FileNotFoundError(p)
        hot_files[n] = p

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    runs = {m: [] for m in methods}
    lock_path = Path.home()/".cache/geoivf/speed-device.lock"
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for m in order:
                    kwargs = {}
                    if m == "navhints":
                        kwargs["ivf"] = args.ivf
                    elif m.startswith("hot-"):
                        n = int(m.split("-")[1])
                        kwargs["hot_ids"] = hot_files[n]
                    rr = run(
                        args.binary,args.out,f"r{rep}-{m}",
                        args.queries,args.gt,args.index_prefix,**kwargs
                    )
                    runs[m].append(rr)
                    save(args.out/"runs.partial.json", runs)
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}

    # A full-node cache necessarily stores the full vector. This lower bound
    # intentionally excludes ALL adjacency/hash/container overhead and thus
    # favors the cache in memory comparisons.
    memory = {}
    for n in HOT_COUNTS:
        vector_lb = n * DIM * FLOAT_BYTES
        memory[f"hot-{n}"] = {
            "nodes": n,
            "full_vector_bytes_lower_bound": vector_lb,
            "full_vector_mib_lower_bound": vector_lb/(1<<20),
            "excluded_from_lower_bound": [
                "adjacency list contents and capacities",
                "AdjacencyList/Vec headers",
                "hash-table entries/control bytes",
                "associated-data array",
                "allocator metadata",
            ],
        }
    ivf_manifest=json.loads(args.ivf.with_suffix(args.ivf.suffix+".manifest.json").read_text())
    nav_payload=args.ivf.stat().st_size+ivf_manifest["nlist"]*(64+4)

    result={
        "workload":"PubMed1M / MedRAG-Zipf frozen heldout 5000",
        "search":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS},
        "training":"same first 5000 traversal queries for hot-cache ranking and NavHints",
        "navhints_runtime_payload_bytes":nav_payload,
        "hot_cache_memory_lower_bounds":memory,
        "memory_guardrail":(
            "hot-cache bytes are vector-only lower bounds; actual hot-cache RAM is strictly larger"
        ),
        "summary":summary,
    }
    save(args.out/"hot-cache-frontier.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
