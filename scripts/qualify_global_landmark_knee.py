#!/usr/bin/env python3
"""Evaluate the global learned-landmark count knee at exact recall."""
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
BUDGETS = (64, 128, 256, 512, 768, 1024, 1536, 2048, 2500)


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


def run_one(binary, out, tag, queries, gt, index_prefix, *, starts=None, router=None, waypoint=None):
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
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)
    if starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(starts)
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
        raise ValueError(f"{tag}: expected one row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0:
        raise ValueError("recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-binary", type=Path, required=True)
    ap.add_argument("--waypoint-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--landmark-dir", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in (
        "global_binary", "waypoint_binary", "queries", "gt", "index_prefix",
        "landmark_dir", "router", "waypoint", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    manifest = json.loads((args.landmark_dir / "global-landmarks.manifest.json").read_text())
    variants = {}
    for name, meta in manifest["variants"].items():
        p = args.landmark_dir / meta["file"]
        if not p.is_file():
            raise FileNotFoundError(p)
        variants[name] = {"path": p, **meta}

    methods = ["ours-fp32-router"]
    for budget in BUDGETS:
        methods += [f"landmarks-b{budget}", f"portals512-plus-landmarks-b{budget}"]

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    if method == "ours-fp32-router":
                        row = run_one(
                            args.waypoint_binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix,
                            router=args.router, waypoint=args.waypoint,
                        )
                    else:
                        row = run_one(
                            args.global_binary, args.out, f"r{rep}-{method}",
                            held, args.gt, args.index_prefix,
                            starts=variants[method]["path"],
                        )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for method in methods:
        rr = [x for x in rows if x["method"] == method]
        s = {
            "rounds": len(rr),
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "median_io_us": med(rr, "mean_io_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }
        if method == "ours-fp32-router":
            s.update({"state_bytes": 3_159_860, "candidate_ids": None})
        else:
            m = variants[method]
            s.update({
                "state_bytes": int(m["state_bytes"]),
                "candidate_ids": int(m.get("combined_unique_ids", m["landmark_ids"])),
            })
        summary[method] = s

    ref = summary["ours-fp32-router"]
    feasible = []
    for name, s in summary.items():
        if name == "ours-fp32-router":
            continue
        if s["recall_percent"] >= ref["recall_percent"] - 0.05 and s["mean_ios"] <= ref["mean_ios"]:
            feasible.append(name)
    best = max(feasible, key=lambda n: summary[n]["median_qps"]) if feasible else None

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "reference": "ours-fp32-router",
        "best_tiny_state_matching_reference_recall_and_io": best,
        "summary": summary,
        "learner": manifest,
    }
    save(args.out / "global-landmark-knee.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
