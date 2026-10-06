#!/usr/bin/env python3
"""Evaluate persistent warm-page co-result hints on top of the frozen SkipDup cache."""
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
BEAM = 8
K = 10
LS = (10, 20, 40, 80, 160, 320)
PERSISTENT_VARIANT = 5

def save(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2) + "\n")

def fbin_shape(p):
    with p.open("rb") as f:
        return struct.unpack("<II", f.read(8))

def suffix_fbin(src, dst, start):
    r, d = fbin_shape(src)
    with src.open("rb") as fi, dst.open("wb") as fo:
        fi.seek(8 + start * d * 4)
        fo.write(struct.pack("<II", r - start, d))
        rem = (r - start) * d * 4
        while rem:
            b = fi.read(min(16 << 20, rem))
            if not b:
                raise ValueError("truncated fbin")
            fo.write(b)
            rem -= len(b)

def suffix_gt(src, dst, start):
    raw = src.read_bytes()
    r, k = struct.unpack("<II", raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II", r - start, k))
        f.write(raw[8 + start * k * 4 :])

def result_rows(o):
    out = []
    if isinstance(o, dict):
        if "search_l" in o and "mean_latency" in o:
            out.append(o)
        else:
            for v in o.values():
                out.extend(result_rows(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(result_rows(v))
    return out

def run_one(binary, out, tag, queries, gt, index_prefix, ivf, entry_file, persistent):
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
    log = out / f"{tag}.log"
    save(inp, cfg)

    env = os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):
            env.pop(n, None)
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    env["DISKANN_VERTEX_HINT_VARIANT"] = "1"
    if entry_file is not None:
        env["DISKANN_START_POINTS_FILE"] = str(entry_file)
    if persistent:
        env["DISKANN_PERSISTENT_HINT_VARIANT"] = str(PERSISTENT_VARIANT)

    with log.open("w") as lf:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rows = sorted(result_rows(json.loads(output.read_text())), key=lambda r: int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS):
        raise ValueError(f"{tag}: bad L grid")
    return rows

def aggregate(reps):
    out = {}
    for l in LS:
        rs = [r for rr in reps for r in rr if int(r["search_l"]) == l]
        out[str(l)] = {
            "recall_percent": float(np.mean([float(r["recall"]) for r in rs])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in rs])),
            "latency_us": float(np.median([float(r["mean_latency"]) for r in rs])),
            "cpu_us": float(np.median([float(r["mean_cpu_time"]) for r in rs])),
            "comparisons": float(np.mean([float(r["mean_comparisons"]) for r in rs])),
        }
    return out

def mono(s):
    out = []
    best = -1e99
    for l in LS:
        r = s[str(l)]["recall_percent"]
        if r + 1e-9 >= best:
            out.append((r, s[str(l)]))
            best = max(best, r)
    return out

def interp(s, t, field):
    p = mono(s)
    if t < p[0][0] or t > p[-1][0]:
        return None
    for (a, ra), (b, rb) in zip(p, p[1:]):
        if a <= t <= b:
            x = 0 if b <= a + 1e-12 else (t - a) / (b - a)
            return float(ra[field]) + x * (float(rb[field]) - float(ra[field]))
    return float(p[-1][1][field])

def main():
    ap = argparse.ArgumentParser()
    for n in (
        "binary", "queries", "gt5000", "index-prefix", "ivf-16k",
        "entry-dir", "cache-manifest", "persistent-manifest", "work", "out",
    ):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    for n in (
        "binary", "queries", "gt5000", "index_prefix", "ivf_16k",
        "entry_dir", "cache_manifest", "persistent_manifest", "work", "out",
    ):
        setattr(args, n, getattr(args, n).resolve())

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    held = args.work / "held.fbin"
    gt = args.work / "held.gt"
    suffix_fbin(args.queries, held, 9000)
    suffix_gt(args.gt5000, gt, 4000)

    cache_meta = json.loads(args.cache_manifest.read_text())
    persistent_meta = json.loads(args.persistent_manifest.read_text())

    methods = {
        "vertex": (None, False),
        "persistent": (None, True),
    }
    scan_us = {"vertex": 0.0, "persistent": 0.0}
    for cap, c in cache_meta["capacities"].items():
        entry = args.entry_dir / c["skipdup"]["file"]
        methods[f"cache_c{cap}"] = (entry, False)
        methods[f"cache_persistent_c{cap}"] = (entry, True)
        scan = float(c["skipdup"]["python_scan_us_per_eval_query"])
        scan_us[f"cache_c{cap}"] = scan
        scan_us[f"cache_persistent_c{cap}"] = scan

    names = list(methods)
    runs = {m: [] for m in names}

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)

    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(names)
                order = names[shift:] + names[:shift]
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for m in order:
                    entry, persistent = methods[m]
                    rr = run_one(
                        args.binary, args.out, f"r{rep}-{m}", held, gt,
                        args.index_prefix, args.ivf_16k, entry, persistent,
                    )
                    runs[m].append(rr)
                    r40 = next(x for x in rr if int(x["search_l"]) == 40)
                    print(
                        f"{m} L40 recall={float(r40['recall']):.3f} "
                        f"io={float(r40['mean_ios']):.2f} "
                        f"lat={float(r40['mean_latency']):.1f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in names}
    base = summary["vertex"]
    fixed = {}
    targets = (35.0, 45.0, 55.0, 65.0, 72.0, 75.0)

    for t in targets:
        bio = interp(base, t, "mean_ios")
        blat = interp(base, t, "latency_us")
        if bio is None:
            continue
        fixed[str(t)] = {}
        for m in names:
            io = interp(summary[m], t, "mean_ios")
            lat = interp(summary[m], t, "latency_us")
            if io is None:
                continue
            fixed[str(t)][m] = {
                "io_change_percent_vs_vertex": 100 * (io / bio - 1),
                "search_latency_change_percent_vs_vertex": 100 * (lat / blat - 1),
                "python_scan_adjusted_latency_change_percent_vs_vertex":
                    100 * ((lat + scan_us[m]) / blat - 1),
            }

    incremental = {}
    for cap in cache_meta["capacities"]:
        cache = f"cache_c{cap}"
        combo = f"cache_persistent_c{cap}"
        rows = []
        for t in targets:
            ts = str(t)
            cio = interp(summary[cache], t, "mean_ios")
            plat = interp(summary[combo], t, "latency_us")
            clat = interp(summary[cache], t, "latency_us")
            pio = interp(summary[combo], t, "mean_ios")
            if None in (cio, pio, clat, plat):
                continue
            rows.append({
                "recall": t,
                "persistent_incremental_io_change_percent": 100 * (pio / cio - 1),
                "persistent_incremental_latency_change_percent": 100 * (plat / clat - 1),
            })
        incremental[cap] = rows

    ranking = []
    for m in names:
        if m == "vertex":
            continue
        vals = [
            fixed[str(t)][m]
            for t in targets
            if str(t) in fixed and m in fixed[str(t)]
        ]
        ranking.append({
            "method": m,
            "mean_io_change_percent_vs_vertex":
                float(np.mean([x["io_change_percent_vs_vertex"] for x in vals])),
            "mean_search_latency_change_percent_vs_vertex":
                float(np.mean([x["search_latency_change_percent_vs_vertex"] for x in vals])),
        })
    ranking.sort(key=lambda x: x["mean_io_change_percent_vs_vertex"])

    result = {
        "policy": (
            "isolate persistence: same SkipDup value cache and same actual prior L160 results; "
            "persistent warm co-results use latest-write-wins 4-ID page overlay"
        ),
        "persistent_payload": persistent_meta,
        "cache_manifest": cache_meta,
        "evaluation": {"Ls": list(LS), "reps": args.reps},
        "summary": summary,
        "fixed_recall": fixed,
        "persistent_increment_on_cache": incremental,
        "ranking": ranking,
    }
    save(args.out / "persistent-coresult.json", result)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
