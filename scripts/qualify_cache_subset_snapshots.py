#!/usr/bin/env python3
"""Evaluate 10-slot hub snapshots filled from existing 512-ID winner cache.

Every arm replays 5,000 queries in chronological order, then measures the
final 1,000. Snapshot arms choose one selected hub after every Xth completed
query, retain its earlier persisted hints, insert that query's successful rank1
winner, and fill unused slots with nearest winners in the existing value cache.

We count exactly one 4-KiB page-write operation per changed published page.
No physical writes occur yet; read-side I/O and CPU are measured inside Rust.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import struct
import subprocess
from pathlib import Path
import numpy as np

LS = (10, 20, 40, 80, 160, 320)
RECALLS = (35., 45., 55., 65., 72., 75.)
METHODS = ("vertex", "onlinehub10_cache512", "snapshot1", "snapshot5", "snapshot10", "snapshot20")
CONFIG = {
    "vertex": (0, 0, 0),
    "onlinehub10_cache512": (10, 512, 0),
    "snapshot1": (10, 512, 1),
    "snapshot5": (10, 512, 5),
    "snapshot10": (10, 512, 10),
    "snapshot20": (10, 512, 20),
}
STATS = re.compile(
    r"ONLINE_SNAPSHOT_STATS L=(\d+) cadence=(\d+) write_ops=(\d+) "
    r"unique_hubs=(\d+) overwrites=(\d+) unchanged=(\d+) winners_written=(\d+) "
    r"warm_writes=(\d+) eval_writes=(\d+) active_pages=(\d+)"
)

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")

def replay_queries(src, dst):
    with src.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
        assert rows == 10000 and dim == 768
        f.seek(8 + 5000 * dim * 4)
        raw = f.read(5000 * dim * 4)
    if len(raw) != 5000 * dim * 4:
        raise ValueError("truncated query input")
    with dst.open("wb") as f:
        f.write(struct.pack("<II", 5000, dim))
        f.write(raw)

def results_tree(o):
    out = []
    if isinstance(o, dict):
        if "search_l" in o and "mean_latency" in o:
            out.append(o)
        else:
            for v in o.values():
                out.extend(results_tree(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(results_tree(v))
    return out

def run(binary, out, tag, q, gt, prefix, ivf, method):
    hub, cache, cadence = CONFIG[method]
    cfg = {"search_directories": [str(out)], "jobs": [{"type": "disk-index", "content": {
        "source": {"disk-index-source": "Load", "data_type": "float32", "load_path": str(prefix)},
        "search_phase": {"queries": str(q), "groundtruth": str(gt),
            "search_list": list(LS), "beam_width": 8, "recall_at": 10, "num_threads": 4,
            "is_flat_search": False, "distance": "inner_product",
            "vector_filters_file": None, "num_nodes_to_cache": None,
            "search_io_limit": None, "post_processor": None}}}]}
    inp = out / (tag + ".input.json")
    output = out / (tag + ".output.json")
    log = out / (tag + ".log")
    save(inp, cfg)

    env = os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):
            env.pop(n, None)
    env.update({
        "DISKANN_HINT_IVF_FILE": str(ivf),
        "DISKANN_HINT_IVF_NPROBE": "8",
        "DISKANN_HINT_IVF_MAX_STARTS": "1",
        "DISKANN_VERTEX_HINT_VARIANT": "1",
        "DISKANN_ONLINE_HUB_REPLAY": "1",
        "DISKANN_ONLINE_HUB_CAPACITY": str(hub),
        "DISKANN_ONLINE_HUB_WARMUP": "4000",
        "DISKANN_ONLINE_HUB_FLUSH_BATCH": "10",
        "DISKANN_COMBINED_VALUE_CACHE_CAPACITY": str(cache),
        "DISKANN_WINNER_SNAPSHOT_CADENCE": str(cadence),
    })
    with log.open("w") as lf:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=lf, stderr=subprocess.STDOUT, env=env, check=True)
    rows = sorted(results_tree(json.loads(output.read_text())), key=lambda x: int(x["search_l"]))
    if [int(x["search_l"]) for x in rows] != list(LS):
        raise ValueError("unexpected L grid for " + tag)
    diag = {}
    for m in STATS.finditer(log.read_text()):
        diag[m.group(1)] = dict(zip(
            ("cadence", "write_ops", "unique_hubs", "overwrites", "unchanged",
             "winners_written", "warm_writes", "eval_writes", "active_pages"),
            map(int, m.groups()[1:])))
    if len(diag) != len(LS):
        raise ValueError("snapshot diagnostics absent for " + tag)
    return rows, diag

def summarize(reps):
    out = {}
    for l in LS:
        rr = [r for runs in reps for r in runs if int(r["search_l"]) == l]
        out[str(l)] = {
            "recall": float(np.mean([float(x["recall"]) for x in rr])),
            "reads": float(np.mean([float(x["mean_ios"]) for x in rr])),
            "latency_us": float(np.median([float(x["mean_latency"]) for x in rr])),
            "cpu_us": float(np.median([float(x["mean_cpu_time"]) for x in rr])),
        }
    return out

def interp(curve, recall, field):
    points = []
    last = -float("inf")
    for l in LS:
        p = curve[str(l)]
        if p["recall"] >= last:
            points.append(p)
            last = p["recall"]
    if recall < points[0]["recall"] or recall > points[-1]["recall"]:
        return None
    for a, b in zip(points, points[1:]):
        if a["recall"] <= recall <= b["recall"]:
            d = b["recall"] - a["recall"]
            t = (recall - a["recall"]) / d if d else 0
            return a[field] + t * (b[field] - a[field])
    return points[-1][field]

def main():
    ap = argparse.ArgumentParser()
    for n in ("binary", "queries", "gt5000", "index-prefix", "ivf-16k", "work", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    for n in ("binary", "queries", "gt5000", "index_prefix", "ivf_16k", "work", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    q = args.work / "replay5000.fbin"
    replay_queries(args.queries, q)
    runs = {m: [] for m in METHODS}
    diag = {m: [] for m in METHODS}
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:4]))
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(METHODS)
                order = METHODS[shift:] + METHODS[:shift]
                print("rep", rep, "order", " ".join(order), flush=True)
                for method in order:
                    values, d = run(args.binary, args.out, f"rep{rep}-{method}",
                                    q, args.gt5000, args.index_prefix,
                                    args.ivf_16k, method)
                    runs[method].append(values)
                    diag[method].append(d)
                    r40 = next(x for x in values if int(x["search_l"]) == 40)
                    print(method, "L40",
                          "recall", round(float(r40["recall"]), 4),
                          "reads", round(float(r40["mean_ios"]), 3),
                          "lat_us", round(float(r40["mean_latency"]), 2),
                          "writes_eval", d["40"]["eval_writes"], flush=True)
    finally:
        os.sched_setaffinity(0, set(affinity))

    summary = {m: summarize(runs[m]) for m in METHODS}
    stats = {}
    for m in METHODS:
        stats[m] = {}
        for l in LS:
            v = [rep[str(l)] for rep in diag[m]]
            stats[m][str(l)] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
            stats[m][str(l)]["writes_per_measured_query"] = stats[m][str(l)]["eval_writes"] / 1000.
    fixed = {}
    for r in RECALLS:
        a = interp(summary["vertex"], r, "reads")
        b = interp(summary["vertex"], r, "latency_us")
        h = interp(summary["onlinehub10_cache512"], r, "reads")
        fixed[str(r)] = {}
        for m in METHODS:
            v = interp(summary[m], r, "reads")
            t = interp(summary[m], r, "latency_us")
            if None in (a, b, h, v, t):
                continue
            fixed[str(r)][m] = {
                "read_delta_vs_vertex_percent": 100 * (v / a - 1),
                "read_delta_vs_combined_percent": 100 * (v / h - 1),
                "latency_delta_vs_vertex_percent": 100 * (t / b - 1),
            }

    result = {
        "definition": "Periodic, one-page published hub snapshot: rank1 current success, then existing published hub entries, then nearest cached successful IDs to fill up to ten slots.",
        "query_lookup": "One real resident-PQ scan of 512 IDs; same scan yields ranked cache candidates.",
        "coherence": "Only previously published pages visible; updates published after completed queries.",
        "write_model": "Each changed published page counts one 4096-byte disk write; write time not charged to latency.",
        "evaluation": {"warm": 4000, "measured": 1000, "reps": args.reps, "Ls": list(LS)},
        "summary": summary, "diagnostics": stats, "fixed_recall": fixed,
    }
    save(args.out / "cache-subset-snapshots.json", result)
    print(json.dumps({"fixed_recall": fixed, "writes": stats,
                      "summary": summary}, indent=2), flush=True)

if __name__ == "__main__":
    main()
