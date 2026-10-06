#!/usr/bin/env python3
"""Runtime screen for in-traversal regional NavHints.

Compares the unchanged canonical 16K NavHints start against the same start plus
an equal-budget per-region navigation table.  The held-out suffix is the final
1000 queries; exact top-16 ground truth is sliced from the existing exact
5000-query suffix truthset.
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
LS = (10, 20, 40, 80, 160)
RUN_TIMEOUT_SECONDS = 600


def save(path: Path, obj):
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
    if not 0 <= start < rows:
        raise ValueError("bad suffix start")
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


def suffix_gt(src: Path, dst: Path, start: int):
    raw = src.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated truthset")
    rows, k = struct.unpack("<II", raw[:8])
    if len(raw) != 8 + rows * k * 4:
        raise ValueError("bad truthset size")
    if not 0 <= start < rows:
        raise ValueError("bad truthset suffix")
    out_rows = rows - start
    lo = 8 + start * k * 4
    with dst.open("wb") as f:
        f.write(struct.pack("<II", out_rows, k))
        f.write(raw[lo:])


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


def run_one(binary, out, tag, queries, gt, index_prefix, start_ivf, regional_file):
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
    logp = out / f"{tag}.log"
    save(inp, cfg)

    env = os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_REGIONAL_HINT_FILE",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_PROGRESSIVE_HINT_TOPK",
        "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)

    env["DISKANN_HINT_IVF_FILE"] = str(start_ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    if regional_file is not None:
        env["DISKANN_REGIONAL_HINT_FILE"] = str(regional_file)

    cmd = [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)]
    started = time.monotonic()
    with logp.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise TimeoutError(tag)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    rows = sorted(result_rows(json.loads(output.read_text())), key=lambda r: int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS):
        raise ValueError(f"{tag}: bad L grid")
    print(f"completed {tag} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return rows


def aggregate(reps):
    out = {}
    for l in LS:
        rows = [r for rr in reps for r in rr if int(r["search_l"]) == l]
        out[str(l)] = {
            "rounds": len(rows),
            "recall_percent": float(np.mean([float(r["recall"]) for r in rows])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in rows])),
            "median_qps": float(np.median([float(r["qps"]) for r in rows])),
            "median_latency_us": float(np.median([float(r["mean_latency"]) for r in rows])),
            "median_cpu_us": float(np.median([float(r["mean_cpu_time"]) for r in rows])),
            "median_io_time_us": float(np.median([float(r["mean_io_time"]) for r in rows])),
            "mean_comparisons": float(np.mean([float(r["mean_comparisons"]) for r in rows])),
            "mean_hops": float(np.mean([float(r["mean_hops"]) for r in rows])),
        }
    return out


def monotone(summary):
    pts = []
    best = -1e99
    for l in LS:
        row = summary[str(l)]
        r = float(row["recall_percent"])
        if r + 1e-9 >= best:
            pts.append((r, row))
            best = max(best, r)
    return pts


def interp(summary, target, field):
    pts = monotone(summary)
    if target < pts[0][0] or target > pts[-1][0]:
        return None
    for (lr, lo), (hr, hi) in zip(pts, pts[1:]):
        if lr <= target <= hr:
            if hr <= lr + 1e-12:
                return float(hi[field])
            a = (target - lr) / (hr - lr)
            return float(lo[field]) + a * (float(hi[field]) - float(lo[field]))
    if target == pts[-1][0]:
        return float(pts[-1][1][field])
    return None


def payload_ivf(path: Path):
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    return path.stat().st_size + int(manifest["nlist"]) * (64 + 4)


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "gt5000", "index-prefix", "ivf-16k", "regional", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    for name in ("binary", "queries", "gt5000", "index_prefix", "ivf_16k", "regional", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected workload")
    held = args.work / "heldout-last1000.fbin"
    gt = args.work / "heldout-last1000.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

    methods = ("canonical16k", "regional256x16")
    runs = {m: [] for m in methods}

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                order = methods[rep % 2:] + methods[:rep % 2]
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for method in order:
                    rr = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, gt, args.index_prefix, args.ivf_16k,
                        args.regional if method == "regional256x16" else None,
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    print(
                        f"finished rep={rep} {method} "
                        f"L20 recall={float(rr[1]['recall']):.3f} "
                        f"io={float(rr[1]['mean_ios']):.3f} "
                        f"cpu={float(rr[1]['mean_cpu_time']):.1f} "
                        f"lat={float(rr[1]['mean_latency']):.1f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}
    same_l = {}
    for l in LS:
        a = summary["canonical16k"][str(l)]
        b = summary["regional256x16"][str(l)]
        same_l[str(l)] = {
            "canonical_recall": a["recall_percent"],
            "regional_recall": b["recall_percent"],
            "recall_delta_points": b["recall_percent"] - a["recall_percent"],
            "canonical_ios": a["mean_ios"],
            "regional_ios": b["mean_ios"],
            "io_change_percent": 100 * (b["mean_ios"] / a["mean_ios"] - 1),
            "canonical_latency_us": a["median_latency_us"],
            "regional_latency_us": b["median_latency_us"],
            "latency_change_percent": 100 * (b["median_latency_us"] / a["median_latency_us"] - 1),
            "canonical_cpu_us": a["median_cpu_us"],
            "regional_cpu_us": b["median_cpu_us"],
            "cpu_change_percent": 100 * (b["median_cpu_us"] / a["median_cpu_us"] - 1),
            "canonical_comparisons": a["mean_comparisons"],
            "regional_comparisons": b["mean_comparisons"],
            "comparison_delta": b["mean_comparisons"] - a["mean_comparisons"],
        }

    fixed = {}
    lo = max(summary[m]["10"]["recall_percent"] for m in methods)
    hi = min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0, 45.0, 55.0, 65.0):
        if not lo <= target <= hi:
            continue
        a = {
            k: interp(summary["canonical16k"], target, k)
            for k in ("mean_ios", "median_latency_us", "median_cpu_us", "mean_comparisons")
        }
        b = {
            k: interp(summary["regional256x16"], target, k)
            for k in ("mean_ios", "median_latency_us", "median_cpu_us", "mean_comparisons")
        }
        fixed[str(target)] = {
            "canonical16k": a,
            "regional256x16": b,
            "regional_vs_canonical": {
                "io_change_percent": 100 * (b["mean_ios"] / a["mean_ios"] - 1),
                "latency_change_percent": 100 * (b["median_latency_us"] / a["median_latency_us"] - 1),
                "cpu_change_percent": 100 * (b["median_cpu_us"] / a["median_cpu_us"] - 1),
                "comparison_change_percent": 100 * (b["mean_comparisons"] / a["mean_comparisons"] - 1),
            },
        }

    result = {
        "workload": "MedRAG-Zipf final 1000 queries; exact PubMed1M IP top-16 truth sliced from frozen suffix GT",
        "training": "first 9000 queries, on-policy canonical 16K NavHints, teacher L=64",
        "regional_policy": {
            "regions": 256,
            "hints_per_region": 16,
            "warmup_entry_position": 8,
            "runtime": "at most one newly entered region scored per natural beam; best unvisited hint offered through fixed-L gate",
        },
        "memory_payload_bytes": {
            "canonical_hint_ivf": payload_ivf(args.ivf_16k),
            "regional_runtime_file": args.regional.stat().st_size,
            "regional_total_incremental": args.regional.stat().st_size,
        },
        "evaluation": {"K": K, "Ls": list(LS), "beam": BEAM, "threads": THREADS, "repetitions": args.reps},
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
    }
    save(args.out / "regional-navhints-runtime.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
