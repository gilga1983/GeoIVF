#!/usr/bin/env python3
"""Realistic Recall@10 curves for flat and indexed NavHints."""
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
LS = (10, 20, 40, 80, 160)


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


def run_method(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    flat_starts: Path | None = None,
    ivf: Path | None = None,
    probe: int | None = None,
):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": list(LS),
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
        "DISKANN_GLOBAL_START_IDS_FILE", "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE", "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT", "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE", "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE", "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    if flat_starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(flat_starts)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = str(probe)
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
        )

    rr = result_rows(json.loads(output.read_text()))
    by_l = {int(r["search_l"]): dict(r) for r in rr}
    if set(by_l) != set(LS):
        raise ValueError(f"{tag}: expected L rows {LS}, got {sorted(by_l)}")
    if any(float(r["recall"]) < 0 for r in by_l.values()):
        raise ValueError(f"{tag}: recall missing")
    return by_l


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


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
    ap.add_argument("--reps", type=int, default=3)
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
    allowed = sorted(os.sched_getaffinity(0))
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
                    kw = {}
                    if method == "flat-b2048":
                        kw["flat_starts"] = args.flat_2048
                    elif method == "ivf-p8":
                        kw.update(ivf=args.ivf, probe=8)
                    elif method == "ivf-p16":
                        kw.update(ivf=args.ivf, probe=16)
                    result = run_method(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, **kw,
                    )
                    for l, row in result.items():
                        rows.append({"rep": rep, "method": method, "L": l, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    ivf_manifest = json.loads(
        args.ivf.with_suffix(args.ivf.suffix + ".manifest.json").read_text()
    )
    summary = {}
    for method in methods:
        summary[method] = {}
        for l in LS:
            rr = [x for x in rows if x["method"] == method and x["L"] == l]
            summary[method][str(l)] = {
                "rounds": len(rr),
                "state_bytes": (
                    0 if method == "baseline-medoid"
                    else args.flat_2048.stat().st_size if method == "flat-b2048"
                    else args.ivf.stat().st_size
                ),
                "recall_at_10_percent": avg(rr, "recall"),
                "mean_ios": avg(rr, "mean_ios"),
                "median_qps": med(rr, "qps"),
                "median_latency_us": med(rr, "mean_latency"),
                "median_cpu_us": med(rr, "mean_cpu_time"),
                "mean_hops": avg(rr, "mean_hops"),
                "mean_comparisons": avg(rr, "mean_comparisons"),
            }

    # For each NavHints arm, compare with the best baseline L not below its recall.
    matched = {}
    baseline = summary["baseline-medoid"]
    for method in ("flat-b2048", "ivf-p8", "ivf-p16"):
        matched[method] = {}
        for l in LS:
            target = summary[method][str(l)]["recall_at_10_percent"]
            candidates = [
                bl for bl in LS
                if baseline[str(bl)]["recall_at_10_percent"] >= target
            ]
            if candidates:
                bl = min(candidates, key=lambda x: baseline[str(x)]["recall_at_10_percent"])
                matched[method][str(l)] = {
                    "navhints_L": l,
                    "baseline_L": bl,
                    "navhints_recall": target,
                    "baseline_recall": baseline[str(bl)]["recall_at_10_percent"],
                    "io_ratio_nav_over_baseline": (
                        summary[method][str(l)]["mean_ios"] / baseline[str(bl)]["mean_ios"]
                    ),
                    "qps_ratio_nav_over_baseline": (
                        summary[method][str(l)]["median_qps"] / baseline[str(bl)]["median_qps"]
                    ),
                }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "evaluation": {"K": K, "Ls": list(LS), "beam": IO_BEAM, "threads": THREADS},
        "ivf": ivf_manifest,
        "summary": summary,
        "recall_matched_against_medoid": matched,
    }
    save(args.out / "hint-ivf-recall-curve.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
