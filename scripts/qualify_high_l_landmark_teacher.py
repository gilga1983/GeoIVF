#!/usr/bin/env python3
"""Deploy landmark dictionaries learned from stronger offline DiskANN teachers.

Every deployed arm runs the identical cheap online search (K=L=1, beam=8).
Only the training trace used to rank landmark IDs changes:
  medoid-start DiskANN at teacher L in {1,4,8,16,32,64}, or portal teacher.
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
TEACHER_LS = (1, 4, 8, 16, 32, 64)
BUDGETS = (1024, 1536, 2048)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
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
            block = fin.read(min(16 << 20, remaining))
            if not block:
                raise ValueError("truncated fbin")
            fout.write(block)
            remaining -= len(block)


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


def run_one(binary, out, tag, queries, gt, index_prefix, starts=None):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [1],
        "beam_width": IO_BEAM,
        "recall_at": 1,
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

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1:
        raise ValueError(f"{tag}: expected one result row")
    row = dict(rows[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: recall missing")
    return row


def avg(rs, key):
    return float(np.mean([float(r[key]) for r in rs]))


def med(rs, key):
    return float(np.median([float(r[key]) for r in rs]))


def load_variant(root: Path, budget: int):
    manifest = json.loads((root / "global-landmarks.manifest.json").read_text())
    meta = manifest["variants"][f"landmarks-b{budget}"]
    p = root / meta["file"]
    if not p.is_file():
        raise FileNotFoundError(p)
    return manifest, {"path": p, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--teachers-root", type=Path, required=True)
    ap.add_argument("--portal-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "teachers_root", "portal_dir", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    variants = {}
    learner_manifests = {}
    for teacher_l in TEACHER_LS:
        root = args.teachers_root / f"L{teacher_l}"
        learner_manifests[f"medoid-L{teacher_l}"] = json.loads((root / "global-landmarks.manifest.json").read_text())
        for budget in BUDGETS:
            _, meta = load_variant(root, budget)
            variants[f"medoid-L{teacher_l}-b{budget}"] = meta
    learner_manifests["portal"] = json.loads((args.portal_dir / "global-landmarks.manifest.json").read_text())
    for budget in BUDGETS:
        _, meta = load_variant(args.portal_dir, budget)
        variants[f"portal-b{budget}"] = meta

    methods = ["baseline-medoid"] + sorted(variants)
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    starts = None if method == "baseline-medoid" else variants[method]["path"]
                    row = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, starts=starts,
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        meta = None if method == "baseline-medoid" else variants[method]
        summary[method] = {
            "rounds": len(rr),
            "state_bytes": 0 if meta is None else int(meta["state_bytes"]),
            "candidate_ids": 0 if meta is None else int(meta["landmark_ids"]),
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "median_io_us": med(rr, "mean_io_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }

    best_by_budget = {}
    for budget in BUDGETS:
        candidates = [f"medoid-L{l}-b{budget}" for l in TEACHER_LS]
        best_by_budget[str(budget)] = {
            "best_recall": max(candidates, key=lambda n: summary[n]["recall_percent"]),
            "best_io": min(candidates, key=lambda n: summary[n]["mean_ios"]),
            "best_qps": max(candidates, key=lambda n: summary[n]["median_qps"]),
            "portal_reference": f"portal-b{budget}",
        }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "online_search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "offline_teacher_ls": list(TEACHER_LS),
        "budgets": list(BUDGETS),
        "landmark_score": "global sum of first expansion positions",
        "summary": summary,
        "best_medoid_teacher_by_budget": best_by_budget,
        "learner_manifests": learner_manifests,
    }
    save(args.out / "high-l-landmark-teacher.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
