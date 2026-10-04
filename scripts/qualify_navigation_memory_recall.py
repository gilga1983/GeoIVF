#!/usr/bin/env python3
"""Exact-recall check for the same-memory navigation comparison.

Uses the cached exact PubMed1M inner-product ground truth for the final 5000
MedRAG-Zipf queries. All arms use k=L=1 and beam=8, matching the throughput
campaign. This is an accuracy validation, not a timing campaign.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
from pathlib import Path


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


def run_one(
    binary,
    out,
    tag,
    queries,
    gt,
    index_prefix,
    *,
    cache_nodes=None,
    qsev=None,
    router=None,
    waypoint=None,
):
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
    env.pop("DISKANN_SKIP_RECALL", None)
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
    if float(row["recall"]) < 0.0:
        raise ValueError(f"{tag}: exact recall was not computed")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsev-binary", type=Path, required=True)
    ap.add_argument("--waypoint-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--qsev", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    args = ap.parse_args()

    for name in (
        "qsev_binary", "waypoint_binary", "queries", "gt", "index_prefix",
        "qsev", "router", "waypoint", "work", "out",
    ):
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

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows_out = {}
            rows_out["medoid"] = run_one(
                args.qsev_binary, args.out, "medoid", held, args.gt, args.index_prefix,
            )
            rows_out["cache-1024"] = run_one(
                args.qsev_binary, args.out, "cache-1024", held, args.gt,
                args.index_prefix, cache_nodes=1024,
            )
            rows_out["qsev"] = run_one(
                args.qsev_binary, args.out, "qsev", held, args.gt,
                args.index_prefix, qsev=args.qsev,
            )
            rows_out["ours-512"] = run_one(
                args.waypoint_binary, args.out, "ours-512", held, args.gt,
                args.index_prefix, router=args.router, waypoint=args.waypoint,
            )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {
        name: {
            "recall_percent": float(row["recall"]),
            "mean_ios": float(row["mean_ios"]),
            "qps": float(row["qps"]),
            "mean_latency_us": float(row["mean_latency"]),
            "mean_hops": float(row["mean_hops"]),
            "mean_comparisons": float(row["mean_comparisons"]),
        }
        for name, row in rows_out.items()
    }
    result = {
        "workload": "MedRAG-Zipf heldout 5000 vs exact PubMed1M IP ground truth",
        "k": K,
        "L": K,
        "beam": IO_BEAM,
        "summary": summary,
        "note": "single exact-recall pass; timing is diagnostic only",
    }
    save(args.out / "navigation-memory-recall-k1.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
