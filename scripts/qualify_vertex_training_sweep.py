#!/usr/bin/env python3
"""Evaluate a packed vertex-NavHint training-size sweep on held-out queries."""
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
FIXED_RECALL_TARGETS = (35.0, 45.0, 55.0, 65.0, 72.0, 75.0)
RUN_TIMEOUT_SECONDS = 900


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin")
    return rows, dim


def suffix_fbin(src: Path, dst: Path, start: int):
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


def suffix_gt(src: Path, dst: Path, start: int):
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


def fixed_metric(base, method, target):
    fields = ("mean_ios", "median_latency_us", "median_cpu_us")
    bvals = [interp(base, target, f) for f in fields]
    mvals = [interp(method, target, f) for f in fields]
    if any(x is None for x in bvals + mvals):
        return None
    bio, blat, bcpu = bvals
    io, lat, cpu = mvals
    return {
        "mean_ios": io,
        "io_change_percent_vs_canonical": 100 * (io / bio - 1),
        "latency_us": lat,
        "latency_change_percent_vs_canonical": 100 * (lat / blat - 1),
        "cpu_us": cpu,
        "cpu_change_percent_vs_canonical": 100 * (cpu / bcpu - 1),
    }


def main():
    ap = argparse.ArgumentParser()
    for name in (
        "binary", "queries", "gt5000", "index-prefix", "ivf-16k",
        "payload-manifest", "embed-manifest", "work", "out",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for name in (
        "binary", "queries", "gt5000", "index_prefix", "ivf_16k",
        "payload_manifest", "embed_manifest", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected workload")

    payload_meta = json.loads(args.payload_manifest.read_text())
    embed_meta = json.loads(args.embed_manifest.read_text())
    if embed_meta["physical_file_size_change"] != 0:
        raise ValueError("training-sweep graph changed physical file size")

    variants = payload_meta["variants"]
    if variants[0]["kind"] != "regional" or variants[0]["index"] != 0:
        raise ValueError("variant zero must be regional baseline")

    method_meta = {
        "canonical16k": {
            "variant_index": None,
            "kind": "canonical",
            "training_queries": None,
            "min_support": None,
        },
        "regional_full9k": {
            "variant_index": 0,
            "kind": "regional",
            "training_queries": variants[0]["training_queries"],
            "min_support": None,
        },
    }
    for v in variants[1:]:
        if v["kind"] != "vertex":
            raise ValueError("unexpected non-vertex sweep arm")
        method_meta[v["name"]] = {
            "variant_index": int(v["index"]),
            "kind": "vertex",
            "training_queries": int(v["training_queries"]),
            "min_support": int(v["min_support"]),
            "coverage": v["coverage"],
            "support_nonzero_vertices": int(v["support_nonzero_vertices"]),
            "support_nonzero_fraction": float(v["support_nonzero_fraction"]),
        }

    methods = tuple(method_meta)
    held = args.work / "heldout-last1000.fbin"
    gt = args.work / "heldout-last1000.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

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
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for method in order:
                    rr = run_one(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        held,
                        gt,
                        args.index_prefix,
                        args.ivf_16k,
                        method_meta[method]["variant_index"],
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    r80 = next(x for x in rr if int(x["search_l"]) == 80)
                    r320 = next(x for x in rr if int(x["search_l"]) == 320)
                    print(
                        f"finished {method} "
                        f"L80 recall={float(r80['recall']):.3f} io={float(r80['mean_ios']):.2f} "
                        f"L320 recall={float(r320['recall']):.3f} io={float(r320['mean_ios']):.2f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}
    base = summary["canonical16k"]

    same_l = {}
    for l in LS:
        b = base[str(l)]
        same_l[str(l)] = {}
        for method in methods:
            r = summary[method][str(l)]
            same_l[str(l)][method] = {
                "recall_percent": r["recall_percent"],
                "recall_delta_points_vs_canonical": r["recall_percent"] - b["recall_percent"],
                "mean_ios": r["mean_ios"],
                "io_change_percent_vs_canonical": 100 * (r["mean_ios"] / b["mean_ios"] - 1),
                "latency_us": r["median_latency_us"],
                "latency_change_percent_vs_canonical": 100 * (
                    r["median_latency_us"] / b["median_latency_us"] - 1
                ),
                "cpu_us": r["median_cpu_us"],
                "cpu_change_percent_vs_canonical": 100 * (
                    r["median_cpu_us"] / b["median_cpu_us"] - 1
                ),
                "comparison_delta_vs_canonical": (
                    r["mean_comparisons"] - b["mean_comparisons"]
                ),
            }

    fixed = {}
    for target in FIXED_RECALL_TARGETS:
        fixed[str(target)] = {
            method: fixed_metric(base, summary[method], target)
            for method in methods
        }

    long_search = {}
    for method in methods:
        rows = [same_l[str(l)][method] for l in (80, 160, 320)]
        long_search[method] = {
            "mean_recall_delta_points_L80_320": float(
                np.mean([r["recall_delta_points_vs_canonical"] for r in rows])
            ),
            "mean_io_change_percent_L80_320": float(
                np.mean([r["io_change_percent_vs_canonical"] for r in rows])
            ),
            "mean_latency_change_percent_L80_320": float(
                np.mean([r["latency_change_percent_vs_canonical"] for r in rows])
            ),
        }

    training_curve = {}
    thresholds = sorted(
        {m["min_support"] for m in method_meta.values() if m["kind"] == "vertex"}
    )
    for threshold in thresholds:
        points = []
        arms = [
            (name, meta) for name, meta in method_meta.items()
            if meta["kind"] == "vertex" and meta["min_support"] == threshold
        ]
        arms.sort(key=lambda x: x[1]["training_queries"])
        for name, meta in arms:
            points.append({
                "method": name,
                "training_queries": meta["training_queries"],
                "specialized_vertices": int(meta["coverage"]["specialized_vertices"]),
                "specialized_fraction": float(meta["coverage"]["specialized_fraction"]),
                "fully_local_vertices": int(meta["coverage"]["fully_local_vertices"]),
                "fully_local_fraction": float(meta["coverage"]["fully_local_fraction"]),
                "support_nonzero_vertices": meta["support_nonzero_vertices"],
                "support_nonzero_fraction": meta["support_nonzero_fraction"],
                "fixed_recall": {
                    str(t): fixed[str(t)][name] for t in FIXED_RECALL_TARGETS
                },
                "long_search": long_search[name],
                "same_L": {
                    str(l): same_l[str(l)][name] for l in LS
                },
            })
        training_curve[str(threshold)] = points

    result = {
        "workload": "MedRAG-Zipf final 1000 queries; exact PubMed1M IP top-16 truth",
        "training_sweep": payload_meta["sampling"],
        "thresholds": payload_meta["thresholds"],
        "policy": {
            "bootstrap": "canonical 16K Hint-IVF",
            "regional_fallback": "fixed 256-region map learned from all 9000 teacher traces",
            "continuation_slots": payload_meta["slots_per_variant"],
            "per_beam": "closest expanded vertex only; at most one fixed-L admission",
        },
        "storage": embed_meta,
        "evaluation": {
            "K": K,
            "Ls": list(LS),
            "beam": BEAM,
            "threads": THREADS,
            "repetitions": args.reps,
            "fixed_recall_note": "linear interpolation between measured L points",
        },
        "method_meta": method_meta,
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
        "long_search_summary": long_search,
        "training_curve": training_curve,
    }
    save(args.out / "vertex-navhints-training-sweep.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
