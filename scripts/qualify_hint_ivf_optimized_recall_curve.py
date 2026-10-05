#!/usr/bin/env python3
"""Optimized Recall@10 curves with dense baseline recall bracketing."""
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
K = 10
NAV_LS = (10, 20, 40, 80, 160)
BASE_LS = (
    10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48,
    56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 176,
)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def suffix_fbin(src: Path, dst: Path, start: int):
    rows, dim = fbin_shape(src)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remaining = (rows - start) * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)


def result_rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            out.append(obj)
        else:
            for v in obj.values():
                out.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(result_rows(v))
    return out


def run_method(binary, out, tag, queries, gt, index_prefix, ls, *, flat=None, ivf=None, probe=None):
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
                    "search_list": list(ls),
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
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE", "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE", "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT", "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE", "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE", "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    if flat is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(flat)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = str(probe)
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
        )
    rows = result_rows(json.loads(output.read_text()))
    by_l = {int(r["search_l"]): dict(r) for r in rows}
    if set(by_l) != set(ls):
        raise ValueError(f"{tag}: expected {ls}, got {sorted(by_l)}")
    return by_l


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def summarize(rows, method, ls, payload):
    out = {}
    for l in ls:
        rr = [r for r in rows if r["method"] == method and r["L"] == l]
        out[str(l)] = {
            "rounds": len(rr),
            "runtime_total_payload_bytes": int(payload),
            "recall_at_10_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_io_us": med(rr, "mean_io_time"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "median_pq_preprocess_us": med(rr, "mean_pq_preprocess_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }
    return out


def bracket_baseline(baseline, target):
    points = sorted(
        [(int(l), v["recall_at_10_percent"], v) for l, v in baseline.items()],
        key=lambda x: x[1],
    )
    lower = None
    upper = None
    for point in points:
        if point[1] <= target:
            lower = point
        if point[1] >= target and upper is None:
            upper = point
    if lower is None:
        lower = points[0]
    if upper is None:
        upper = points[-1]
    return lower, upper


def interpolate(lower, upper, target, field):
    lrec, urec = lower[1], upper[1]
    if abs(urec - lrec) < 1e-12:
        return float(upper[2][field])
    w = (target - lrec) / (urec - lrec)
    w = min(1.0, max(0.0, w))
    return float(lower[2][field]) + w * (float(upper[2][field]) - float(lower[2][field]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--flat-2048", type=Path, required=True)
    ap.add_argument("--ivf", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    for name in ("binary","queries","gt","index_prefix","flat_2048","ivf","work","out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    methods = ("baseline-medoid", "flat-b2048", "ivf-p8", "ivf-p16")
    rows = []
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    if method == "baseline-medoid":
                        ls, kw = BASE_LS, {}
                    elif method == "flat-b2048":
                        ls, kw = NAV_LS, {"flat": args.flat_2048}
                    elif method == "ivf-p8":
                        ls, kw = NAV_LS, {"ivf": args.ivf, "probe": 8}
                    else:
                        ls, kw = NAV_LS, {"ivf": args.ivf, "probe": 16}
                    result = run_method(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, ls, **kw,
                    )
                    for l, row in result.items():
                        rows.append({"rep": rep, "method": method, "L": l, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    ivf_manifest = json.loads(args.ivf.with_suffix(args.ivf.suffix + ".manifest.json").read_text())
    ivf_payload = args.ivf.stat().st_size + 512 * (64 + 4)

    summary = {
        "baseline-medoid": summarize(rows, "baseline-medoid", BASE_LS, 0),
        "flat-b2048": summarize(rows, "flat-b2048", NAV_LS, args.flat_2048.stat().st_size),
        "ivf-p8": summarize(rows, "ivf-p8", NAV_LS, ivf_payload),
        "ivf-p16": summarize(rows, "ivf-p16", NAV_LS, ivf_payload),
    }

    comparisons = {}
    baseline = summary["baseline-medoid"]
    for method in ("flat-b2048", "ivf-p8", "ivf-p16"):
        comparisons[method] = {}
        for l in NAV_LS:
            nav = summary[method][str(l)]
            target = nav["recall_at_10_percent"]
            low, high = bracket_baseline(baseline, target)
            interpolated_io = interpolate(low, high, target, "mean_ios")
            interpolated_latency = interpolate(low, high, target, "median_latency_us")
            comparisons[method][str(l)] = {
                "navhints_L": l,
                "navhints_recall": target,
                "baseline_lower": {
                    "L": low[0], "recall": low[1],
                    "mean_ios": low[2]["mean_ios"],
                    "median_qps": low[2]["median_qps"],
                },
                "baseline_upper": {
                    "L": high[0], "recall": high[1],
                    "mean_ios": high[2]["mean_ios"],
                    "median_qps": high[2]["median_qps"],
                },
                "interpolated_baseline_mean_ios": interpolated_io,
                "interpolated_baseline_latency_us": interpolated_latency,
                "io_ratio_nav_over_interpolated_baseline": nav["mean_ios"] / interpolated_io,
                "latency_ratio_nav_over_interpolated_baseline": (
                    nav["median_latency_us"] / interpolated_latency
                ),
                # Conservative throughput ratio against the higher-recall bracket.
                "qps_ratio_nav_over_higher_recall_baseline": (
                    nav["median_qps"] / high[2]["median_qps"]
                ),
            }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "implementation": "packed-direct 16K hints, spherical 512 lists",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "evaluation": {
            "K": K,
            "navhints_Ls": list(NAV_LS),
            "baseline_Ls": list(BASE_LS),
            "beam": IO_BEAM,
            "threads": THREADS,
            "reps": args.reps,
        },
        "ivf": ivf_manifest,
        "summary": summary,
        "recall_bracketed_comparisons": comparisons,
    }
    save(args.out / "optimized-recall10-curve.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
