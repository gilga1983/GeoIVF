#!/usr/bin/env python3
"""Matched-recall confirmation for the strongest Starling MemGraph baseline.

Evaluates the fixed 3 MiB frequency-trained Starling MemGraph at mem_L={2,3}
against the 512-region learned-waypoint design. All SSD searches use the same
PubMed1M DiskANN graph, PQ codes, k=L=1, beam=8, four query threads, exact
heldout ground truth, and three order-rotated repetitions.

Starling navigation time is accounted conservatively by serial composition:
combined_qps = 1 / (1/nav_qps + 1/disk_qps)
combined_latency = nav_latency + disk_latency.
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


def run_one(binary, out, tag, queries, gt, index_prefix, *, starts=None, router=None, waypoint=None):
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
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)
    if starts is not None:
        env["DISKANN_START_POINTS_FILE"] = str(starts)
    if router is not None:
        env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
        env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    if waypoint is not None:
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(waypoint)
        env["DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"] = "40"

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
        raise ValueError(f"{tag}: exact recall missing")
    return row


def raw_metrics(row):
    return {
        "recall_percent": float(row["recall"]),
        "mean_ios": float(row["mean_ios"]),
        "qps": float(row["qps"]),
        "mean_latency_us": float(row["mean_latency"]),
        "mean_hops": float(row["mean_hops"]),
        "mean_comparisons": float(row["mean_comparisons"]),
    }


def mem_metrics(row, nav):
    disk = raw_metrics(row)
    nq = float(nav["nav_qps"])
    dq = disk["qps"]
    return {
        "recall_percent": disk["recall_percent"],
        "mean_ios": disk["mean_ios"],
        "disk_qps": dq,
        "nav_qps": nq,
        "combined_qps": 1.0 / (1.0 / nq + 1.0 / dq),
        "disk_mean_latency_us": disk["mean_latency_us"],
        "nav_mean_latency_us": float(nav["nav_mean_latency_us"]),
        "combined_mean_latency_us": disk["mean_latency_us"] + float(nav["nav_mean_latency_us"]),
        "mean_hops": disk["mean_hops"],
        "mean_comparisons": disk["mean_comparisons"],
    }


def ours_metrics(row):
    r = raw_metrics(row)
    return {
        "recall_percent": r["recall_percent"],
        "mean_ios": r["mean_ios"],
        "combined_qps": r["qps"],
        "combined_mean_latency_us": r["mean_latency_us"],
        "mean_hops": r["mean_hops"],
        "mean_comparisons": r["mean_comparisons"],
    }


def average(rows):
    keys = rows[0].keys()
    return {
        key: float(np.mean([float(r[key]) for r in rows]))
        for key in keys
        if all(isinstance(r[key], (int, float)) for r in rows)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-binary", type=Path, required=True)
    ap.add_argument("--waypoint-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    args = ap.parse_args()

    for name in ("start_binary","waypoint_binary","queries","gt","index_prefix","router","waypoint","manifest","work","out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if (rows, dim, args.train_rows) != (10000, 768, 5000):
        raise ValueError("expected frozen 10000x768, split 5000/5000")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    manifest = json.loads(args.manifest.read_text())
    specs = {}
    for l in (2, 3):
        nav = manifest["results"][str(l)]
        starts = Path(nav["start_file"]).resolve()
        if not starts.is_file():
            raise FileNotFoundError(starts)
        specs[f"frequency-L{l}"] = (starts, nav)

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    accum = {"ours-512": [], "frequency-L2": [], "frequency-L3": []}
    raw = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            methods = list(accum)
            for rep in range(3):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    if method == "ours-512":
                        row = run_one(
                            args.waypoint_binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix,
                            router=args.router, waypoint=args.waypoint,
                        )
                        m = ours_metrics(row)
                    else:
                        starts, nav = specs[method]
                        row = run_one(
                            args.start_binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix, starts=starts,
                        )
                        m = mem_metrics(row, nav)
                    accum[method].append(m)
                    raw.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", raw)
    finally:
        os.sched_setaffinity(0, set(allowed))

    confirmed = {k: average(v) for k, v in accum.items()}
    ours = confirmed["ours-512"]
    deltas = {}
    for method in ("frequency-L2", "frequency-L3"):
        x = confirmed[method]
        deltas[method] = {
            "recall_delta_ours_minus_memgraph_points": ours["recall_percent"] - x["recall_percent"],
            "io_reduction_ours_vs_memgraph_fraction": 1.0 - ours["mean_ios"] / x["mean_ios"],
            "qps_speedup_ours_vs_memgraph": ours["combined_qps"] / x["combined_qps"],
            "latency_reduction_ours_vs_memgraph_fraction": 1.0 - ours["combined_mean_latency_us"] / x["combined_mean_latency_us"],
        }

    result = {
        "comparison": "same-budget frequency-trained Starling MemGraph matched-recall probe",
        "memgraph_bytes": manifest["deployed_index_bytes_on_disk"],
        "memgraph_sample_count": manifest["sample_count"],
        "three_run_means": confirmed,
        "deltas": deltas,
    }
    save(args.out / "memgraph-matched-recall.json", result)
    save(args.out / "rows.json", raw)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
