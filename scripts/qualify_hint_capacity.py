#!/usr/bin/env python3
"""Capacity sweep for ID-only Hint-IVF.

Compares:
  * flat PQ scan over each stored hint budget;
  * fixed 256-list Hint-IVF, p=8;
  * scaled-directory Hint-IVF, p=8, where nlist grows with capacity to keep
    bucket occupancy roughly constant.

All arms deploy the same ordinary DiskANN search (K=L=1, beam=8). Routing and
search costs are both included in QueryStatistics by patch_diskann_hint_ivf.py.
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
NPROBE = 8
CAPACITIES = (2048, 4096, 8192, 16000, 20949)
SCALED_NLIST = {
    2048: 32,
    4096: 64,
    8192: 128,
    16000: 256,
    20949: 384,
}


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


def run_one(binary, out, tag, queries, gt, index_prefix, *, flat_starts=None, hint_ivf=None):
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

    if flat_starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(flat_starts)
    if hint_ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(hint_ivf)
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
    rr = result_rows(json.loads(output.read_text()))
    if len(rr) != 1:
        raise ValueError(f"{tag}: expected one result row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--landmark-dir", type=Path, required=True)
    ap.add_argument("--ivf-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "landmark_dir", "ivf_dir", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    flats = {}
    fixed = {}
    scaled = {}
    manifests = {}
    for cap in CAPACITIES:
        flat = args.landmark_dir / f"landmarks-b{cap}.bin"
        if not flat.is_file():
            raise FileNotFoundError(flat)
        flats[cap] = flat

        fixed_path = args.ivf_dir / f"hints-b{cap}-nlist256.bin"
        if not fixed_path.is_file():
            raise FileNotFoundError(fixed_path)
        fixed[cap] = fixed_path
        manifests[f"fixed-b{cap}"] = json.loads(
            fixed_path.with_suffix(fixed_path.suffix + ".manifest.json").read_text()
        )

        nlist = SCALED_NLIST[cap]
        scaled_path = args.ivf_dir / f"hints-b{cap}-nlist{nlist}.bin"
        if not scaled_path.is_file():
            raise FileNotFoundError(scaled_path)
        scaled[cap] = scaled_path
        manifests[f"scaled-b{cap}"] = json.loads(
            scaled_path.with_suffix(scaled_path.suffix + ".manifest.json").read_text()
        )

    methods = ["baseline-medoid"]
    methods += [f"flat-b{cap}" for cap in CAPACITIES]
    methods += [f"fixed-b{cap}" for cap in CAPACITIES]
    methods += [
        f"scaled-b{cap}"
        for cap in CAPACITIES
        if SCALED_NLIST[cap] != 256
    ]

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
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    kwargs = {}
                    if method.startswith("flat-b"):
                        cap = int(method.split("b", 1)[1])
                        kwargs["flat_starts"] = flats[cap]
                    elif method.startswith("fixed-b"):
                        cap = int(method.split("b", 1)[1])
                        kwargs["hint_ivf"] = fixed[cap]
                    elif method.startswith("scaled-b"):
                        cap = int(method.split("b", 1)[1])
                        kwargs["hint_ivf"] = scaled[cap]
                    row = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, **kwargs,
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)

    summary = {}
    for method in methods:
        rr = [x for x in rows if x["method"] == method]
        if method == "baseline-medoid":
            state_bytes = 0
            capacity = 0
            nlist = 0
            kind = "baseline"
        else:
            cap = int(method.split("b", 1)[1])
            capacity = cap
            if method.startswith("flat-"):
                state_bytes = flats[cap].stat().st_size
                nlist = 0
                kind = "flat"
            elif method.startswith("fixed-"):
                state_bytes = fixed[cap].stat().st_size
                nlist = 256
                kind = "fixed"
            else:
                state_bytes = scaled[cap].stat().st_size
                nlist = SCALED_NLIST[cap]
                kind = "scaled"
        summary[method] = {
            "rounds": len(rr),
            "kind": kind,
            "capacity": capacity,
            "nlist": nlist,
            "nprobe": 0 if kind in ("baseline", "flat") else NPROBE,
            "state_bytes": state_bytes,
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
            "mean_hops": avg(rr, "mean_hops"),
        }

    # Alias the identical 16K fixed/scaled point so plots can treat both curves uniformly.
    summary["scaled-b16000"] = {
        **summary["fixed-b16000"],
        "kind": "scaled",
        "nlist": 256,
    }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "online_search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "capacity_values": list(CAPACITIES),
        "fixed_config": {"nlist": 256, "nprobe": NPROBE},
        "scaled_nlist": {str(k): v for k, v in SCALED_NLIST.items()},
        "scaled_nprobe": NPROBE,
        "eligible_vertices": 20949,
        "manifests": manifests,
        "summary": summary,
    }
    save(args.out / "hint-capacity-results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
