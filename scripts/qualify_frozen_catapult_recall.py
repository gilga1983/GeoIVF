#!/usr/bin/env python3
"""Recall validation for frozen Catapult vs cap-40 adaptive caches.

Catapult states are trained on queries [0,5000), frozen, then all methods are
evaluated once on queries [5000,10000) using exact PubMed1M inner-product GT.
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
SEEDS = (0, 1, 2)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin")
    return rows, dim


def slice_fbin(src: Path, dst: Path, start: int, count: int) -> Path:
    rows, dim = fbin_shape(src)
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError("invalid fbin slice")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", count, dim))
        remaining = count * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)
    return dst


def result_rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "recall" in obj:
            out.append(obj)
        else:
            for v in obj.values():
                out.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(result_rows(v))
    return out


def cfg(queries: Path, gt: Path, index_prefix: Path):
    return {
        "search_directories": [str(queries.parent)],
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


def run(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    router: Path,
    *,
    skip_recall: bool,
    seed: int | None = None,
    portal: bool = False,
    waypoint_cache: Path | None = None,
    waypoint_max_ids: int | None = None,
    snapshot_load: Path | None = None,
    snapshot_dump: Path | None = None,
    freeze_catapult: bool = False,
):
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg(queries, gt, index_prefix))
    env = os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_CATAPULT_SNAPSHOT_LOAD",
        "DISKANN_CATAPULT_SNAPSHOT_DUMP",
        "DISKANN_CATAPULT_FREEZE",
    ):
        env.pop(name, None)

    if skip_recall:
        env["DISKANN_SKIP_RECALL"] = "1"
    if portal:
        env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
        env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    if waypoint_cache is not None:
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(waypoint_cache)
    if waypoint_max_ids is not None:
        env["DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"] = str(waypoint_max_ids)
    if seed is not None:
        env["DISKANN_PAPER_CATAPULT"] = "1"
        env["DISKANN_CATAPULT_HASHES"] = "8"
        env["DISKANN_CATAPULT_CAPACITY"] = "40"
        env["DISKANN_CATAPULT_SEED"] = str(seed)
    if snapshot_load is not None:
        env["DISKANN_CATAPULT_SNAPSHOT_LOAD"] = str(snapshot_load)
    if snapshot_dump is not None:
        env["DISKANN_CATAPULT_SNAPSHOT_DUMP"] = str(snapshot_dump)
    if freeze_catapult:
        env["DISKANN_CATAPULT_FREEZE"] = "1"

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
        raise ValueError(f"{tag}: expected one row, got {len(rr)}")
    row = dict(rr[0])
    if skip_recall and float(row["recall"]) != -1.0:
        raise ValueError("expected skipped-recall sentinel")
    if not skip_recall and not (0.0 <= float(row["recall"]) <= 100.0):
        raise ValueError("invalid recall")
    return row


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
    args = ap.parse_args()
    for attr in (
        "binary", "queries", "gt", "index_prefix", "portal_router",
        "cache_dir", "work", "out",
    ):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if (rows, dim) != (10000, 768):
        raise ValueError(f"expected 10000x768 queries, got {rows}x{dim}")

    # Validate GT header for the heldout suffix.
    with args.gt.open("rb") as f:
        n_gt, k_gt = struct.unpack("<II", f.read(8))
    if n_gt != 5000 or k_gt < K:
        raise ValueError(f"GT shape must be 5000 x >=1, got {n_gt}x{k_gt}")

    train = slice_fbin(args.queries, args.work / "train.fbin", 0, 5000)
    held = slice_fbin(args.queries, args.work / "heldout.fbin", 5000, 5000)

    endpoint = args.cache_dir / "global-endpoint-s1.bin"
    waypoint = args.cache_dir / "waypoint-s1.bin"
    for p in (endpoint, waypoint, args.portal_router, args.gt):
        if not p.is_file():
            raise FileNotFoundError(p)

    snapshots = {}
    for seed in SEEDS:
        for portal in (False, True):
            kind = "portal-catapult" if portal else "catapult"
            snap = args.work / f"{kind}-seed{seed}.snapshot"
            print(f"TRAIN {kind} seed={seed}", flush=True)
            run(
                args.binary, args.out, f"train-{kind}-s{seed}",
                train, args.gt, args.index_prefix, args.portal_router,
                skip_recall=True,
                seed=seed,
                portal=portal,
                snapshot_dump=snap,
            )
            snapshots[(kind, seed)] = snap

    methods = [
        "portal",
        "global-endpoint-cap40",
        "waypoint-cap40",
        *[f"catapult-s{x}" for x in SEEDS],
        *[f"portal-catapult-s{x}" for x in SEEDS],
    ]

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for method in methods:
                print(f"EVAL {method}", flush=True)
                if method == "portal":
                    kwargs = dict(portal=True)
                elif method == "global-endpoint-cap40":
                    kwargs = dict(
                        portal=True,
                        waypoint_cache=endpoint,
                        waypoint_max_ids=40,
                    )
                elif method == "waypoint-cap40":
                    kwargs = dict(
                        portal=True,
                        waypoint_cache=waypoint,
                        waypoint_max_ids=40,
                    )
                elif method.startswith("portal-catapult-s"):
                    seed = int(method.rsplit("s", 1)[1])
                    kwargs = dict(
                        seed=seed,
                        portal=True,
                        snapshot_load=snapshots[("portal-catapult", seed)],
                        freeze_catapult=True,
                    )
                elif method.startswith("catapult-s"):
                    seed = int(method.rsplit("s", 1)[1])
                    kwargs = dict(
                        seed=seed,
                        portal=False,
                        snapshot_load=snapshots[("catapult", seed)],
                        freeze_catapult=True,
                    )
                else:
                    raise AssertionError(method)
                results[method] = run(
                    args.binary, args.out, f"eval-{method}",
                    held, args.gt, args.index_prefix, args.portal_router,
                    skip_recall=False,
                    **kwargs,
                )
    finally:
        os.sched_setaffinity(0, set(allowed))

    def seed_avg(prefix: str, key: str) -> float:
        return float(np.mean([float(results[f"{prefix}-s{s}"][key]) for s in SEEDS]))

    summary = {
        method: {
            "recall": float(row["recall"]),
            "mean_ios": float(row["mean_ios"]),
            "qps": float(row["qps"]),
            "mean_latency_us": float(row["mean_latency"]),
        }
        for method, row in results.items()
    }
    aggregate = {
        "catapult": {
            "mean_recall": seed_avg("catapult", "recall"),
            "mean_ios": seed_avg("catapult", "mean_ios"),
        },
        "portal-catapult": {
            "mean_recall": seed_avg("portal-catapult", "recall"),
            "mean_ios": seed_avg("portal-catapult", "mean_ios"),
        },
    }
    out = {
        "workload": "MedRAG-Zipf heldout 5000 vs PubMed1M exact IP GT",
        "k": K,
        "L": K,
        "adaptive_cached_ids_per_query_cap": 40,
        "summary": summary,
        "seed_aggregate": aggregate,
    }
    save(args.out / "frozen-catapult-recall-k1.json", out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
