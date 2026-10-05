#!/usr/bin/env python3
"""Exact-recall ablation of the traversal teacher used to learn global landmarks.

Compares identical skip-position landmark learners trained from:
  * portal-seeded DiskANN traversals; and
  * ordinary medoid-start DiskANN traversals.

Deployment is identical in all learned arms: only a global uint32 ID list is
retained and query-specific selection reuses DiskANN's resident PQ codes.
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
BUDGETS = (256, 512, 768, 1024, 1536, 2048, 2500)


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
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated source fbin")
            fout.write(b)
            remaining -= len(b)
    return n, dim


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
        raise ValueError(f"{tag}: expected one result row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0.0:
        raise ValueError(f"{tag}: exact recall was not computed")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def load_variants(root: Path):
    manifest = json.loads((root / "global-landmarks.manifest.json").read_text())
    out = {}
    for budget in BUDGETS:
        name = f"landmarks-b{budget}"
        meta = manifest["variants"][name]
        path = root / meta["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        out[budget] = {"path": path, **meta}
    return manifest, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--portal-dir", type=Path, required=True)
    ap.add_argument("--medoid-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "portal_dir", "medoid_dir", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    nheld, _ = suffix_fbin(args.queries, held, 5000)
    if nheld != 5000:
        raise AssertionError("heldout split mismatch")

    portal_manifest, portal = load_variants(args.portal_dir)
    medoid_manifest, medoid = load_variants(args.medoid_dir)

    methods = ["baseline-medoid"]
    for budget in BUDGETS:
        methods.extend([f"portal-b{budget}", f"medoid-b{budget}"])

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
                    if method == "baseline-medoid":
                        starts = None
                    else:
                        teacher, raw_budget = method.split("-b", 1)
                        budget = int(raw_budget)
                        starts = portal[budget]["path"] if teacher == "portal" else medoid[budget]["path"]
                    row = run_one(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        held,
                        args.gt,
                        args.index_prefix,
                        starts=starts,
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)

    summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        state_bytes = 0
        candidate_ids = 0
        if method != "baseline-medoid":
            teacher, raw_budget = method.split("-b", 1)
            budget = int(raw_budget)
            meta = portal[budget] if teacher == "portal" else medoid[budget]
            state_bytes = int(meta["state_bytes"])
            candidate_ids = int(meta["landmark_ids"])
        summary[method] = {
            "rounds": len(rr),
            "state_bytes": state_bytes,
            "candidate_ids": candidate_ids,
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "median_io_us": med(rr, "mean_io_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }

    paired = {}
    for budget in BUDGETS:
        p = summary[f"portal-b{budget}"]
        m = summary[f"medoid-b{budget}"]
        paired[str(budget)] = {
            "recall_delta_medoid_minus_portal_points": m["recall_percent"] - p["recall_percent"],
            "io_delta_medoid_minus_portal": m["mean_ios"] - p["mean_ios"],
            "qps_ratio_medoid_over_portal": m["median_qps"] / p["median_qps"],
            "latency_ratio_medoid_over_portal": m["median_latency_us"] / p["median_latency_us"],
        }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "landmark_score": "global sum of first expansion positions",
        "deployment": "identical global ID-only PQ-scored start pool",
        "budgets": list(BUDGETS),
        "summary": summary,
        "paired_teacher_deltas": paired,
        "portal_teacher_manifest": portal_manifest,
        "medoid_teacher_manifest": medoid_manifest,
    }
    save(args.out / "landmark-teacher-ablation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
