#!/usr/bin/env python3
"""Same-epoch, current-budget competitor frontier for NavHints.

All methods use the same frozen PubMed1M / MedRAG-Zipf heldout 5K queries,
DiskANN graph, PQ state, SSD, beam width, thread count, and dense L grid.

Methods:
* baseline: ordinary DiskANN medoid start
* bfs-30 / bfs-32: native DiskANN full-node BFS cache
* hot-30 / hot-32: full-node cache chosen from the same 5K training history
* qsev-32: DiskANN++-style query-sensitive entry pool, 32 FP32 vectors
* navhints: progressive 16K hints / 512 spherical lists / nprobe=8; route once, retain runner-ups, and admit them after natural beam boundaries
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

THREADS = 4
BEAM = 8
K = 10
LS = (
    10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48,
    56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 176, 192,
    224, 256, 288, 320,
)
NAV_ANCHORS = (10, 20, 40, 80, 160)
CACHE_COUNTS = (30, 32)
DIM = 768
FLOAT_BYTES = 4
MAX_DEGREE = 64
RUN_TIMEOUT_SECONDS = 600
HEARTBEAT_SECONDS = 30


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated fbin header: {path}")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size mismatch: {path}")
    return rows, dim


def result_rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            out.append(obj)
        else:
            for value in obj.values():
                out.extend(result_rows(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(result_rows(value))
    return out


def run_one(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    cache_nodes: int | None = None,
    hot_ids: Path | None = None,
    qsev: Path | None = None,
    ivf: Path | None = None,
    progressive_hints: bool = False,
):
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
                    "num_nodes_to_cache": cache_nodes,
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
    for name in (
        "DISKANN_SKIP_RECALL",
        "DISKANN_STATIC_CACHE_IDS_FILE",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
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

    if hot_ids is not None:
        env["DISKANN_STATIC_CACHE_IDS_FILE"] = str(hot_ids)
    if qsev is not None:
        env["DISKANN_QSEV_FILE"] = str(qsev)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = "8"
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
        if progressive_hints:
            env["DISKANN_PROGRESSIVE_HINTS"] = "1"

    method_log = out / f"{tag}.log"
    command = [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)]
    started = time.monotonic()
    print(
        f"starting tag={tag} binary={binary.name} timeout={RUN_TIMEOUT_SECONDS}s",
        flush=True,
    )
    with method_log.open("w") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        while True:
            try:
                return_code = proc.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                load = os.getloadavg()
                log_bytes = method_log.stat().st_size if method_log.exists() else 0
                print(
                    f"benchmark-heartbeat tag={tag} pid={proc.pid} "
                    f"elapsed={elapsed:.0f}s log_bytes={log_bytes} "
                    f"load1={load[0]:.2f} load5={load[1]:.2f} load15={load[2]:.2f}",
                    flush=True,
                )
                if elapsed >= RUN_TIMEOUT_SECONDS:
                    print(
                        f"timeout tag={tag}; terminating pid={proc.pid}",
                        flush=True,
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise TimeoutError(
                        f"{tag} exceeded {RUN_TIMEOUT_SECONDS}s; see {method_log}"
                    )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    print(
        f"completed tag={tag} elapsed={time.monotonic() - started:.1f}s "
        f"log_bytes={method_log.stat().st_size}",
        flush=True,
    )
    rows = sorted(result_rows(json.loads(output.read_text())), key=lambda r: int(r["search_l"]))
    got = [int(row["search_l"]) for row in rows]
    if got != list(LS):
        raise ValueError(f"{tag}: expected Ls {list(LS)}, got {got}")
    return rows


def aggregate(repetitions):
    by_l = {l: [] for l in LS}
    for rows in repetitions:
        for row in rows:
            by_l[int(row["search_l"])].append(row)

    out = {}
    for l in LS:
        rows = by_l[l]
        if not rows:
            raise ValueError(f"missing L={l}")
        out[str(l)] = {
            "rounds": len(rows),
            "recall_percent": float(np.mean([float(r["recall"]) for r in rows])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in rows])),
            "median_qps": float(np.median([float(r["qps"]) for r in rows])),
            "median_latency_us": float(np.median([float(r["mean_latency"]) for r in rows])),
            "median_io_us": float(np.median([float(r["mean_io_time"]) for r in rows])),
            "mean_cache_hit_percent": float(
                np.mean([float(r["cache_hit_percentage"]) for r in rows])
            ),
            "mean_hops": float(np.mean([float(r["mean_hops"]) for r in rows])),
            "mean_comparisons": float(
                np.mean([float(r["mean_comparisons"]) for r in rows])
            ),
        }
    return out


def monotone_points(summary):
    points = []
    best = -float("inf")
    for l in LS:
        row = summary[str(l)]
        recall = float(row["recall_percent"])
        if recall + 1e-9 >= best:
            points.append({
                "L": l,
                "recall_percent": recall,
                "mean_ios": float(row["mean_ios"]),
                "median_latency_us": float(row["median_latency_us"]),
                "median_qps": float(row["median_qps"]),
            })
            best = max(best, recall)
    return points


def interpolate_at_recall(summary, target):
    pts = monotone_points(summary)
    if target <= pts[0]["recall_percent"]:
        p = pts[0]
        return {
            "mode": "minimum-L already at or above target",
            "lower": None,
            "upper": p,
            "interpolated_mean_ios": p["mean_ios"],
            "interpolated_latency_us": p["median_latency_us"],
        }

    for lower, upper in zip(pts, pts[1:]):
        lo = lower["recall_percent"]
        hi = upper["recall_percent"]
        if lo <= target <= hi:
            alpha = 1.0 if hi <= lo + 1e-12 else (target - lo) / (hi - lo)
            return {
                "mode": "linear interpolation in recall",
                "lower": lower,
                "upper": upper,
                "alpha": alpha,
                "interpolated_mean_ios": (
                    lower["mean_ios"] + alpha * (upper["mean_ios"] - lower["mean_ios"])
                ),
                "interpolated_latency_us": (
                    lower["median_latency_us"]
                    + alpha * (upper["median_latency_us"] - lower["median_latency_us"])
                ),
            }
    return None


def build_matched(summary):
    nav = summary["navhints"]
    competitors = [m for m in summary if m != "navhints"]
    out = {}
    for l in NAV_ANCHORS:
        navrow = nav[str(l)]
        target = float(navrow["recall_percent"])
        item = {
            "navhints_L": l,
            "navhints_recall_percent": target,
            "navhints_mean_ios": float(navrow["mean_ios"]),
            "navhints_latency_us": float(navrow["median_latency_us"]),
            "competitors": {},
        }
        for method in competitors:
            matched = interpolate_at_recall(summary[method], target)
            if matched is None:
                item["competitors"][method] = {
                    "available": False,
                    "reason": "competitor curve does not reach target recall",
                }
                continue
            comp_io = matched["interpolated_mean_ios"]
            comp_lat = matched["interpolated_latency_us"]
            matched["available"] = True
            matched["io_ratio_nav_over_competitor"] = (
                float(navrow["mean_ios"]) / comp_io
            )
            matched["latency_ratio_nav_over_competitor"] = (
                float(navrow["median_latency_us"]) / comp_lat
            )
            matched["io_saving_percent"] = 100.0 * (
                1.0 - float(navrow["mean_ios"]) / comp_io
            )
            matched["latency_saving_percent"] = 100.0 * (
                1.0 - float(navrow["median_latency_us"]) / comp_lat
            )
            item["competitors"][method] = matched
        out[str(l)] = item
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-binary", type=Path, required=True)
    ap.add_argument("--qsev-binary", type=Path, required=True)
    ap.add_argument("--nav-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--hot-dir", type=Path, required=True)
    ap.add_argument("--qsev", type=Path, required=True)
    ap.add_argument("--ivf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in (
        "hot_binary", "qsev_binary", "nav_binary", "queries", "gt",
        "index_prefix", "hot_dir", "qsev", "ivf", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if (rows, dim) != (5000, DIM):
        raise ValueError(f"expected heldout 5000x{DIM} queries, got {rows}x{dim}")

    hot_files = {}
    for n in CACHE_COUNTS:
        p = args.hot_dir / f"hot-cache-n{n}.bin"
        if not p.is_file():
            raise FileNotFoundError(p)
        hot_files[n] = p

    ivf_manifest_path = args.ivf.with_suffix(args.ivf.suffix + ".manifest.json")
    if not ivf_manifest_path.is_file():
        raise FileNotFoundError(ivf_manifest_path)
    ivf_manifest = json.loads(ivf_manifest_path.read_text())
    nlist = int(ivf_manifest["nlist"])
    nav_runtime_bytes = args.ivf.stat().st_size + nlist * (64 + 4)
    qsev_bytes = args.qsev.stat().st_size
    if qsev_bytes > nav_runtime_bytes:
        raise ValueError(
            f"QSEV state {qsev_bytes} exceeds NavHints runtime state {nav_runtime_bytes}"
        )

    methods = [
        "baseline", "bfs-30", "bfs-32", "hot-30", "hot-32", "qsev-32", "navhints"
    ]
    runs = {m: [] for m in methods}

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need at least four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                print(f"rep {rep}: {' '.join(order)}", flush=True)
                for method in order:
                    kwargs = {}
                    binary = args.hot_binary
                    if method.startswith("bfs-"):
                        kwargs["cache_nodes"] = int(method.split("-", 1)[1])
                    elif method.startswith("hot-"):
                        n = int(method.split("-", 1)[1])
                        kwargs["hot_ids"] = hot_files[n]
                    elif method == "qsev-32":
                        binary = args.qsev_binary
                        kwargs["qsev"] = args.qsev
                    elif method == "navhints":
                        binary = args.nav_binary
                        kwargs["ivf"] = args.ivf
                        kwargs["progressive_hints"] = True

                    rr = run_one(
                        binary,
                        args.out,
                        f"r{rep}-{method}",
                        args.queries,
                        args.gt,
                        args.index_prefix,
                        **kwargs,
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    print(
                        f"finished rep={rep} method={method} "
                        f"L10 recall={float(rr[0]['recall']):.3f} "
                        f"ios={float(rr[0]['mean_ios']):.3f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}

    cache_memory = {}
    for n in CACHE_COUNTS:
        vectors = n * DIM * FLOAT_BYTES
        max_edges = n * MAX_DEGREE * 4
        payload = vectors + max_edges
        cache_memory[str(n)] = {
            "nodes": n,
            "vector_payload_bytes": vectors,
            "max_degree_edge_id_payload_bytes": max_edges,
            "vector_plus_max_degree_edge_id_payload_bytes": payload,
            "payload_fraction_of_navhints_state": payload / nav_runtime_bytes,
            "bytes_remaining_after_vector_payload": nav_runtime_bytes - vectors,
            "edge_ids_per_node_affordable_after_vectors": (
                (nav_runtime_bytes - vectors) / (4 * n)
            ),
            "excluded_from_payload_accounting": [
                "hash-table storage/control bytes",
                "AdjacencyList/Vec headers and capacity slack",
                "associated-data array",
                "allocator metadata",
            ],
        }

    result = {
        "workload": "PubMed1M / MedRAG-Zipf frozen heldout final 5000 queries",
        "training": "same first 5000 traversal queries for hot-cache ranking and NavHints",
        "search": {
            "K": K,
            "Ls": list(LS),
            "navhints_anchor_Ls": list(NAV_ANCHORS),
            "beam": BEAM,
            "threads": THREADS,
            "metric": "inner_product",
            "repetitions": args.reps,
            "same_graph_pq_ssd": True,
        },
        "state_budget": {
            "navhints_runtime_payload_bytes": nav_runtime_bytes,
            "navhints_ivf_file_bytes": args.ivf.stat().st_size,
            "navhints_packed_coarse_runtime_bytes": nlist * (64 + 4),
            "qsev_32_bytes": qsev_bytes,
            "qsev_budget_fraction": qsev_bytes / nav_runtime_bytes,
            "full_node_cache": cache_memory,
        },
        "guardrails": [
            "NavHints uses the progressive retained-shortlist policy: the 16K directory is routed once, runner-ups are retained in query-local scratch, and later admission occurs only after natural beam boundaries.",
            "QSEV routing is inside the timed query path.",
            "QSEV uses exact centroid representatives, strengthening the published approximate offline mapping.",
            "The 30-node cache fits vector plus worst-case 64 edge-ID payload under the NavHints state budget before container metadata.",
            "The 32-node cache is deliberately generous: vector payload alone consumes almost the entire NavHints budget.",
            "Cache container/hash/allocator overhead is excluded, which favors the cache baselines.",
            "All methods are order-rotated within one device-locked timing epoch.",
        ],
        "summary": summary,
        "matched_recall_against_navhints": build_matched(summary),
    }
    save(args.out / "paper-competitors-phase1.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
