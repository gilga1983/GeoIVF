#!/usr/bin/env python3
"""Frozen Catapult vs demand-adaptive endpoint/waypoint caches on held-out MedRAG-Zipf.

Training history is queries [0, 5000); evaluation is queries [5000, 10000).
For Catapult, the exact mutable bucket contents are learned on the training prefix,
serialized, then reloaded with updates disabled on the held-out suffix.

Primary cache-policy comparison:
  portal + frozen Catapult
  portal + global demand-weighted endpoints
  portal + global I/O-aware waypoints

Also reports paper Catapult without the static portal and the static portal alone.
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
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin byte size")
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
            block = fin.read(min(16 << 20, remaining))
            if not block:
                raise ValueError("truncated fbin")
            fout.write(block)
            remaining -= len(block)
    return dst


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


def snapshot_entries(path: Path) -> int:
    raw = path.read_bytes()
    if len(raw) < 32 or raw[:8] != b"GICAT001":
        raise ValueError(f"bad Catapult snapshot: {path}")
    return struct.unpack("<I", raw[28:32])[0]


def config(queries: Path, gt: Path, index_prefix: Path) -> dict:
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
    save(inp, config(queries, gt, index_prefix))

    env = os.environ.copy()
    env["DISKANN_SKIP_RECALL"] = "1"
    for name in (
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

    if portal:
        env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
        env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    if waypoint_cache is not None:
        if not portal:
            raise ValueError("waypoint cache requires portal routing")
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(waypoint_cache)
    if waypoint_max_ids is not None:
        if waypoint_cache is None or waypoint_max_ids <= 0:
            raise ValueError("waypoint_max_ids requires a cache and must be positive")
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
    if len(rr) != 1 or int(rr[0]["search_l"]) != K:
        raise ValueError(f"{tag}: unexpected output rows")
    row = dict(rr[0])
    if float(row["recall"]) != -1.0:
        raise ValueError(f"{tag}: recall sentinel missing")
    return row


def mean(rows, key):
    return float(np.mean([float(x[key]) for x in rows]))


def median(rows, key):
    return float(np.median([float(x[key]) for x in rows]))


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
    ap.add_argument("--eval-rows", type=int, default=5000)
    args = ap.parse_args()

    for attr in (
        "binary", "queries", "gt", "index_prefix", "portal_router",
        "cache_dir", "work", "out",
    ):
        setattr(args, attr, getattr(args, attr).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != args.train_rows + args.eval_rows or dim != 768:
        raise ValueError(
            f"expected {args.train_rows + args.eval_rows}x768 queries, got {rows}x{dim}"
        )

    train = slice_fbin(args.queries, args.work / "train-prefix.fbin", 0, args.train_rows)
    held = slice_fbin(
        args.queries,
        args.work / "heldout-suffix.fbin",
        args.train_rows,
        args.eval_rows,
    )

    endpoint = args.cache_dir / "global-endpoint-s1.bin"
    waypoint = args.cache_dir / "waypoint-s1.bin"
    for p in (args.portal_router, endpoint, waypoint):
        if not p.is_file():
            raise FileNotFoundError(p)

    # Train and freeze exact Catapult states under both base-start conditions.
    snapshots = {}
    snapshot_meta = []
    for seed in SEEDS:
        for portal in (False, True):
            kind = "portal-catapult" if portal else "catapult"
            snap = args.work / f"{kind}-seed{seed}.snapshot"
            print(f"TRAIN {kind} seed={seed}", flush=True)
            run(
                args.binary,
                args.out,
                f"train-{kind}-seed{seed}",
                train,
                args.gt,
                args.index_prefix,
                args.portal_router,
                seed=seed,
                portal=portal,
                snapshot_dump=snap,
            )
            if not snap.is_file():
                raise FileNotFoundError(f"snapshot not written: {snap}")
            entries = snapshot_entries(snap)
            snapshots[(kind, seed)] = snap
            snapshot_meta.append({
                "method": kind,
                "seed": seed,
                "resident_entries_after_5000": entries,
                "max_budget_entries": 256 * 40,
            })
            print(f"SNAPSHOT {kind} seed={seed} entries={entries}", flush=True)

    methods = [
        "portal",
        "global-endpoint-s1",
        "global-endpoint-s1-cap40",
        "waypoint-s1",
        "waypoint-s1-cap40",
    ]
    methods += [f"catapult-s{seed}" for seed in SEEDS]
    methods += [f"portal-catapult-s{seed}" for seed in SEEDS]

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs available")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    all_rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)

            # Three rotated passes. Catapult seed-specific arms are independent
            # methods; static methods are repeated for timing robustness.
            for rep in range(3):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    tag = f"eval-r{rep}-{method}"
                    print(f"RUN {tag}", flush=True)
                    kwargs = {}
                    if method == "portal":
                        kwargs = dict(portal=True)
                    elif method == "global-endpoint-s1":
                        kwargs = dict(portal=True, waypoint_cache=endpoint)
                    elif method == "global-endpoint-s1-cap40":
                        kwargs = dict(
                            portal=True,
                            waypoint_cache=endpoint,
                            waypoint_max_ids=40,
                        )
                    elif method == "waypoint-s1":
                        kwargs = dict(portal=True, waypoint_cache=waypoint)
                    elif method == "waypoint-s1-cap40":
                        kwargs = dict(
                            portal=True,
                            waypoint_cache=waypoint,
                            waypoint_max_ids=40,
                        )
                    elif method.startswith("catapult-s"):
                        seed = int(method.rsplit("s", 1)[1])
                        kwargs = dict(
                            seed=seed,
                            portal=False,
                            snapshot_load=snapshots[("catapult", seed)],
                            freeze_catapult=True,
                        )
                    elif method.startswith("portal-catapult-s"):
                        seed = int(method.rsplit("s", 1)[1])
                        kwargs = dict(
                            seed=seed,
                            portal=True,
                            snapshot_load=snapshots[("portal-catapult", seed)],
                            freeze_catapult=True,
                        )
                    else:
                        raise AssertionError(method)

                    row = run(
                        args.binary,
                        args.out,
                        tag,
                        held,
                        args.gt,
                        args.index_prefix,
                        args.portal_router,
                        **kwargs,
                    )
                    all_rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", all_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", all_rows)

    summary = {}
    for method in methods:
        rr = [x for x in all_rows if x["method"] == method]
        summary[method] = {
            "runs": len(rr),
            "mean_ios": mean(rr, "mean_ios"),
            "mean_hops": mean(rr, "mean_hops"),
            "median_qps": median(rr, "qps"),
            "median_latency_us": median(rr, "mean_latency"),
            "median_io_time_us": median(rr, "mean_io_time"),
            "median_cpu_time_us": median(rr, "mean_cpu_time"),
            "catapult_usage_percent": mean(rr, "catapult_usage_percentage"),
            "mean_catapult_starts": mean(rr, "mean_catapult_starts"),
        }

    def aggregate_seed_family(prefix: str):
        rr = []
        for seed in SEEDS:
            rr.append(summary[f"{prefix}-s{seed}"])
        return {
            "seeds": list(SEEDS),
            "mean_ios_across_seed_methods": float(np.mean([x["mean_ios"] for x in rr])),
            "median_of_seed_median_qps": float(np.median([x["median_qps"] for x in rr])),
            "median_of_seed_median_latency_us": float(
                np.median([x["median_latency_us"] for x in rr])
            ),
            "mean_catapult_starts": float(np.mean([x["mean_catapult_starts"] for x in rr])),
        }

    portal_ios = summary["portal"]["mean_ios"]
    for method in (
        "global-endpoint-s1",
        "global-endpoint-s1-cap40",
        "waypoint-s1",
        "waypoint-s1-cap40",
    ):
        summary[method]["io_reduction_vs_portal"] = (
            1.0 - summary[method]["mean_ios"] / portal_ios
        )

    result = {
        "workload": "reconstructed MedRAG-Zipf",
        "split": {
            "training_prefix_queries": args.train_rows,
            "heldout_suffix_queries": args.eval_rows,
        },
        "search": {
            "k": K,
            "search_l": K,
            "beam_width": IO_BEAM,
            "threads": THREADS,
            "recall": "skipped in I/O-policy comparison",
        },
        "catapult": {
            "hashes": 8,
            "bucket_capacity": 40,
            "max_budget_entries": 256 * 40,
            "snapshot_states": snapshot_meta,
            "updates_on_heldout": False,
        },
        "learned_cache_entries": {
            "global_endpoint_s1_file_bytes": endpoint.stat().st_size,
            "waypoint_s1_file_bytes": waypoint.stat().st_size,
        },
        "summary": summary,
        "seed_aggregates": {
            "catapult": aggregate_seed_family("catapult"),
            "portal-catapult": aggregate_seed_family("portal-catapult"),
        },
    }
    save(args.out / "frozen-catapult-heldout-k1-result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
