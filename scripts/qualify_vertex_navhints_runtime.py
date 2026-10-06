#!/usr/bin/env python3
"""Evaluate vertex-granular embedded NavHints, emphasizing long searches."""
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
LS = (10, 20, 40, 80, 160, 320)
RUN_TIMEOUT_SECONDS = 900

METHODS = (
    "canonical16k",
    "regional4",
    "vertex_s2",
    "vertex_s4",
    "vertex_s8",
    "vertex_s16",
)
VARIANT = {
    "canonical16k": None,
    "regional4": 0,
    "vertex_s2": 1,
    "vertex_s4": 2,
    "vertex_s8": 3,
    "vertex_s16": 4,
}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path):
    with path.open("rb") as f:
        raw = f.read(8)
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin")
    return rows, dim


def suffix_fbin(src, dst, start):
    rows, dim = fbin_shape(src)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remain = (rows - start) * dim * 4
        while remain:
            b = fin.read(min(16 << 20, remain))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remain -= len(b)


def suffix_gt(src, dst, start):
    raw = src.read_bytes()
    rows, k = struct.unpack("<II", raw[:8])
    if len(raw) != 8 + rows * k * 4:
        raise ValueError("bad gt")
    with dst.open("wb") as f:
        f.write(struct.pack("<II", rows - start, k))
        f.write(raw[8 + start * k * 4:])


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


def run_one(binary, out, tag, queries, gt, index_prefix, ivf, variant):
    cfg = {
        "search_directories": [str(out)],
        "jobs": [{"type": "disk-index", "content": {
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
        }}],
    }
    inp = out / f"{tag}.input.json"
    output = out / f"{tag}.output.json"
    logp = out / f"{tag}.log"
    save(inp, cfg)

    env = os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL", "DISKANN_HINT_IVF_FILE", "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS", "DISKANN_VERTEX_HINT_VARIANT",
        "DISKANN_EMBEDDED_HINTS", "DISKANN_REGIONAL_HINT_FILE",
        "DISKANN_PROGRESSIVE_HINTS", "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_GLOBAL_START_IDS_FILE", "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE", "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    if variant is not None:
        env["DISKANN_VERTEX_HINT_VARIANT"] = str(variant)

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
            "median_latency_us": float(np.median([float(r["mean_latency"]) for r in rows])),
            "median_cpu_us": float(np.median([float(r["mean_cpu_time"]) for r in rows])),
            "median_qps": float(np.median([float(r["qps"]) for r in rows])),
            "mean_comparisons": float(np.mean([float(r["mean_comparisons"]) for r in rows])),
            "mean_hops": float(np.mean([float(r["mean_hops"]) for r in rows])),
        }
    return out


def monotone(summary):
    pts, best = [], -1e99
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
    if target == pts[0][0]:
        return float(pts[0][1][field])
    for (lr, lo), (hr, hi) in zip(pts, pts[1:]):
        if lr <= target <= hr:
            if hr <= lr + 1e-12:
                return float(hi[field])
            a = (target - lr) / (hr - lr)
            return float(lo[field]) + a * (float(hi[field]) - float(lo[field]))
    return float(pts[-1][1][field])


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "gt5000", "index-prefix", "ivf-16k", "payload-manifest", "embed-manifest", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for name in ("binary", "queries", "gt5000", "index_prefix", "ivf_16k", "payload_manifest", "embed_manifest", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected workload")

    payload_meta = json.loads(args.payload_manifest.read_text())
    embed_meta = json.loads(args.embed_manifest.read_text())
    if embed_meta["physical_file_size_change"] != 0:
        raise ValueError("vertex-hint graph changed physical file size")

    held = args.work / "heldout-last1000.fbin"
    gt = args.work / "heldout-last1000.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

    runs = {m: [] for m in METHODS}
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
                shift = rep % len(METHODS)
                order = METHODS[shift:] + METHODS[:shift]
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for method in order:
                    rr = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, gt, args.index_prefix, args.ivf_16k, VARIANT[method],
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    r160 = next(x for x in rr if int(x["search_l"]) == 160)
                    r320 = next(x for x in rr if int(x["search_l"]) == 320)
                    print(
                        f"finished {method} "
                        f"L160 recall={float(r160['recall']):.3f} io={float(r160['mean_ios']):.2f} "
                        f"L320 recall={float(r320['recall']):.3f} io={float(r320['mean_ios']):.2f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in METHODS}
    base = summary["canonical16k"]

    same_l = {}
    for l in LS:
        b = base[str(l)]
        same_l[str(l)] = {}
        for method in METHODS:
            r = summary[method][str(l)]
            same_l[str(l)][method] = {
                "recall_percent": r["recall_percent"],
                "recall_delta_points_vs_canonical": r["recall_percent"] - b["recall_percent"],
                "mean_ios": r["mean_ios"],
                "io_change_percent_vs_canonical": 100 * (r["mean_ios"] / b["mean_ios"] - 1),
                "latency_us": r["median_latency_us"],
                "latency_change_percent_vs_canonical": 100 * (r["median_latency_us"] / b["median_latency_us"] - 1),
                "cpu_us": r["median_cpu_us"],
                "cpu_change_percent_vs_canonical": 100 * (r["median_cpu_us"] / b["median_cpu_us"] - 1),
                "comparison_delta_vs_canonical": r["mean_comparisons"] - b["mean_comparisons"],
            }

    lo = max(summary[m]["10"]["recall_percent"] for m in METHODS)
    hi = min(summary[m]["320"]["recall_percent"] for m in METHODS)
    fixed = {}
    for target in (35.0, 45.0, 55.0, 65.0, 72.0, 75.0):
        if not lo <= target <= hi:
            continue
        fixed[str(target)] = {}
        bio = interp(base, target, "mean_ios")
        blat = interp(base, target, "median_latency_us")
        bcpu = interp(base, target, "median_cpu_us")
        for method in METHODS:
            io = interp(summary[method], target, "mean_ios")
            lat = interp(summary[method], target, "median_latency_us")
            cpu = interp(summary[method], target, "median_cpu_us")
            fixed[str(target)][method] = {
                "mean_ios": io,
                "io_change_percent_vs_canonical": 100 * (io / bio - 1),
                "latency_us": lat,
                "latency_change_percent_vs_canonical": 100 * (lat / blat - 1),
                "cpu_us": cpu,
                "cpu_change_percent_vs_canonical": 100 * (cpu / bcpu - 1),
            }

    long_search = {}
    for method in METHODS:
        rows = [same_l[str(l)][method] for l in (80, 160, 320)]
        long_search[method] = {
            "mean_recall_delta_points_L80_320": float(np.mean([r["recall_delta_points_vs_canonical"] for r in rows])),
            "mean_io_change_percent_L80_320": float(np.mean([r["io_change_percent_vs_canonical"] for r in rows])),
            "mean_latency_change_percent_L80_320": float(np.mean([r["latency_change_percent_vs_canonical"] for r in rows])),
        }

    result = {
        "workload": "MedRAG-Zipf final 1000 queries; exact PubMed1M IP top-16 truth",
        "training": {
            "queries": 9000,
            "source": "canonical 16K NavHints on-policy L64 traversals",
            "min_vertex_training_position": payload_meta["min_training_position"],
            "score": payload_meta["score"],
            "coverage": payload_meta["coverage"],
        },
        "policy": {
            "bootstrap": "canonical 16K Hint-IVF",
            "continuation_slots": 4,
            "variants": payload_meta["variant_semantics"],
            "per_beam": "closest expanded vertex only; at most one fixed-L admission",
        },
        "storage": embed_meta,
        "evaluation": {
            "K": K,
            "Ls": list(LS),
            "beam": BEAM,
            "threads": THREADS,
            "repetitions": args.reps,
        },
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
        "long_search_summary": long_search,
    }
    save(args.out / "vertex-navhints-runtime.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
