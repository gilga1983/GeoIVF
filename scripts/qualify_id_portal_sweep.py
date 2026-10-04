#!/usr/bin/env python3
"""Evaluate ID-only portal quantity and global learned-start pools.

All arms use the final 5000 MedRAG-Zipf queries, exact PubMed1M ground truth,
the same pinned DiskANN graph/PQ files, K=L=1, beam=8, and four threads.

Global-ID arms score every candidate through DiskANN's existing in-memory PQ
codes. Current 512-region FP32 router + regional waypoint cache is rerun as the
reference arm in the same SSD timing epoch.
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


def run_one(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    global_starts: Path | None = None,
    router: Path | None = None,
    waypoint: Path | None = None,
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
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)

    if global_starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(global_starts)
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
        raise ValueError(f"{tag}: recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-binary", type=Path, required=True)
    ap.add_argument("--waypoint-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in (
        "global_binary", "waypoint_binary", "queries", "gt", "index_prefix",
        "pool_dir", "router", "waypoint", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if (rows, dim, args.train_rows) != (10000, 768, 5000):
        raise ValueError("expected frozen 10000x768 workload and 5000/5000 split")
    held = args.work / "heldout.fbin"
    nheld, _ = suffix_fbin(args.queries, held, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    manifest = json.loads((args.pool_dir / "id-portal-pools.manifest.json").read_text())
    variants = {}
    for name, meta in manifest["variants"].items():
        path = args.pool_dir / meta["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        variants[name] = {"path": path, **meta}

    # Stable order: portal-only then portal+learned at increasing global quantity.
    methods = ["ours-fp32-router"]
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        methods.append(f"portals-n{n}")
        methods.append(f"portals-n{n}-plus-waypoints")
    if set(methods[1:]) != set(variants):
        raise ValueError("pool manifest variants do not match expected sweep")

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    all_rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    if method == "ours-fp32-router":
                        row = run_one(
                            args.waypoint_binary,
                            args.out,
                            f"r{rep}-{method}",
                            held,
                            args.gt,
                            args.index_prefix,
                            router=args.router,
                            waypoint=args.waypoint,
                        )
                    else:
                        row = run_one(
                            args.global_binary,
                            args.out,
                            f"r{rep}-{method}",
                            held,
                            args.gt,
                            args.index_prefix,
                            global_starts=variants[method]["path"],
                        )
                    all_rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)
    summary = {}
    for method in methods:
        rr = [r for r in all_rows if r["method"] == method]
        s = {
            "rounds": len(rr),
            "recall_percent_mean": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_io_us": med(rr, "mean_io_time"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }
        if method == "ours-fp32-router":
            s.update({
                "runtime_state_bytes": 3_159_860,
                "runtime_state_mib": 3_159_860 / (1 << 20),
                "representation": "512 FP32 centers + 512 FP32 portal vectors + regional learned IDs",
            })
        else:
            meta = variants[method]
            s.update({
                "runtime_state_bytes": int(meta["state_bytes"]),
                "runtime_state_mib": float(meta["state_mib"]),
                "candidate_start_ids": int(meta.get("combined_unique_ids", meta["portal_ids"])),
                "portal_ids": int(meta["portal_ids"]),
                "representation": (
                    "uint32 IDs only; all candidates PQ-scored globally using DiskANN resident PQ"
                ),
            })
        summary[method] = s

    result = {
        "workload": "MedRAG-Zipf heldout suffix, exact PubMed1M IP ground truth",
        "search": {"K": K, "L": K, "beam": IO_BEAM, "threads": THREADS},
        "question": (
            "Can portal quantity plus existing PQ scoring replace the 3 MiB full-precision router?"
        ),
        "pool_manifest": manifest,
        "summary": summary,
    }
    save(args.out / "id-portal-sweep.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
