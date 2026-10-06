#!/usr/bin/env python3
"""Evaluate vanilla, one-shot, and progressive frozen NavHints on a public heldout split."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

import numpy as np

THREADS = 4
BEAM = 8
K = 10
LS = (10, 20, 40, 80, 160, 320)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


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


def run_method(binary, out, tag, queries, gt, index_prefix, data_type, distance, ivf=None, progressive=False):
    cfg = {
        "search_directories": [str(out)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": data_type,
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
                    "distance": distance,
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
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
    ):
        env.pop(name, None)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = "8"
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
        if progressive:
            env["DISKANN_PROGRESSIVE_HINTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rr = sorted(result_rows(json.loads(output.read_text())), key=lambda x: int(x["search_l"]))
    if [int(x["search_l"]) for x in rr] != list(LS):
        raise ValueError(f"{tag}: unexpected L rows")
    return [dict(x) for x in rr]


def mean_rows(reps):
    by_l = {}
    for rows in reps:
        for r in rows:
            by_l.setdefault(int(r["search_l"]), []).append(r)
    result = []
    for lval in LS:
        rr = by_l[lval]
        result.append({
            "L": lval,
            "recall_percent": float(np.mean([float(x["recall"]) for x in rr])),
            "mean_ios": float(np.mean([float(x["mean_ios"]) for x in rr])),
            "median_qps": float(np.median([float(x["qps"]) for x in rr])),
            "median_latency_us": float(np.median([float(x["mean_latency"]) for x in rr])),
            "median_cpu_us": float(np.median([float(x["mean_cpu_time"]) for x in rr])),
            "median_pq_preprocess_us": float(
                np.median([float(x["mean_pq_preprocess_time"]) for x in rr])
            ),
            "mean_comparisons": float(np.mean([float(x["mean_comparisons"]) for x in rr])),
            "mean_hops": float(np.mean([float(x["mean_hops"]) for x in rr])),
        })
    return result


def interpolate_baseline(curve, target_recall):
    pts = sorted(curve, key=lambda x: x["recall_percent"])
    for a, b in zip(pts, pts[1:]):
        ra, rb = a["recall_percent"], b["recall_percent"]
        if ra <= target_recall <= rb and rb > ra:
            t = (target_recall - ra) / (rb - ra)
            def lerp(key):
                return a[key] + t * (b[key] - a[key])
            return {
                "bracket_L": [a["L"], b["L"]],
                "recall_percent": target_recall,
                "mean_ios": lerp("mean_ios"),
                "qps": lerp("median_qps"),
                "latency_us": lerp("median_latency_us"),
            }
        if target_recall == ra:
            return {
                "bracket_L": [a["L"], a["L"]],
                "recall_percent": target_recall,
                "mean_ios": a["mean_ios"],
                "qps": a["median_qps"],
                "latency_us": a["median_latency_us"],
            }
    if pts and target_recall == pts[-1]["recall_percent"]:
        p = pts[-1]
        return {
            "bracket_L": [p["L"], p["L"]],
            "recall_percent": target_recall,
            "mean_ios": p["mean_ios"],
            "qps": p["median_qps"],
            "latency_us": p["median_latency_us"],
        }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--dataset-manifest", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--ivf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for n in ("binary", "dataset_manifest", "index_prefix", "ivf", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.dataset_manifest.read_text())
    queries = Path(manifest["files"]["heldout5000"])
    gt = Path(manifest["files"]["heldout5000_gt"])
    if not queries.is_file() or not gt.is_file():
        raise FileNotFoundError("heldout files missing")

    methods = ("baseline", "canonical", "progressive")
    runs = {m: [] for m in methods}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                order = methods if rep % 2 == 0 else tuple(reversed(methods))
                for method in order:
                    rows = run_method(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        queries,
                        gt,
                        args.index_prefix,
                        manifest["data_type"],
                        manifest["metric"],
                        args.ivf if method != "baseline" else None,
                        progressive=(method == "progressive"),
                    )
                    runs[method].append(rows)
                    save(args.out / "runs.partial.json", runs)
    finally:
        os.sched_setaffinity(0, set(allowed))

    baseline = mean_rows(runs["baseline"])
    canonical = mean_rows(runs["canonical"])
    navhints = mean_rows(runs["progressive"])

    same_l = []
    matched = []
    b_by_l = {x["L"]: x for x in baseline}
    c_by_l = {x["L"]: x for x in canonical}
    for n in navhints:
        b = b_by_l[n["L"]]
        c = c_by_l[n["L"]]
        same_l.append({
            "L": n["L"],
            "recall_gain_points_vs_baseline": n["recall_percent"] - b["recall_percent"],
            "io_ratio_vs_baseline": n["mean_ios"] / b["mean_ios"],
            "qps_ratio_vs_baseline": n["median_qps"] / b["median_qps"],
            "recall_gain_points_vs_canonical": n["recall_percent"] - c["recall_percent"],
            "io_ratio_vs_canonical": n["mean_ios"] / c["mean_ios"],
            "latency_ratio_vs_canonical": n["median_latency_us"] / c["median_latency_us"],
        })
        ib = interpolate_baseline(baseline, n["recall_percent"])
        if ib is not None:
            matched.append({
                "navhints_L": n["L"],
                "recall_percent": n["recall_percent"],
                "baseline_bracket_L": ib["bracket_L"],
                "navhints_ios": n["mean_ios"],
                "baseline_interpolated_ios": ib["mean_ios"],
                "io_ratio": n["mean_ios"] / ib["mean_ios"],
                "navhints_qps": n["median_qps"],
                "baseline_interpolated_qps": ib["qps"],
                "qps_ratio": n["median_qps"] / ib["qps"],
            })

    ivf_manifest = json.loads(
        args.ivf.with_suffix(args.ivf.suffix + ".manifest.json").read_text()
    )
    runtime_payload = args.ivf.stat().st_size + ivf_manifest["nlist"] * (64 + 4)

    result = {
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "data_type": manifest["data_type"],
        "distance": manifest["metric"],
        "training": {
            "queries": 5000,
            "teacher_L": 4,
            "hints": ivf_manifest["landmark_ids"],
            "nlist": ivf_manifest["nlist"],
            "nprobe": 8,
        },
        "runtime_payload_bytes": runtime_payload,
        "search": {"K": K, "Ls": list(LS), "beam": BEAM, "threads": THREADS},
        "baseline": baseline,
        "canonical_navhints": canonical,
        "navhints": navhints,
        "navhints_policy": "progressive retained runner-ups after natural beam boundaries; no second routing pass",
        "same_L": same_l,
        "matched_recall_interpolation": matched,
    }
    save(args.out / "public-navhints.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
