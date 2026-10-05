#!/usr/bin/env python3
"""High-repetition confirmation of the optimized Hint-IVF frontier."""
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
CONFIGS = (
    ("sph256-p4", 256, 4),
    ("sph256-p8", 256, 8),
    ("sph512-p4", 512, 4),
    ("sph512-p8", 512, 8),
    ("sph512-p16", 512, 16),
    ("sph1024-p16", 1024, 16),
    ("sph1024-p32", 1024, 32),
)


def save(path: Path, obj) -> None:
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
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remaining = (rows - start) * dim * 4
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


def run_one(binary, out, tag, queries, gt, index_prefix, *, flat=None, ivf=None, nprobe=None):
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
                },
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
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    if flat is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(flat)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = str(nprobe)
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

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
        raise ValueError(f"{tag}: expected one row")
    return dict(rr[0])


def avg(rows, key):
    return float(np.mean([float(x[key]) for x in rows]))


def med(rows, key):
    return float(np.median([float(x[key]) for x in rows]))


def p25(rows, key):
    return float(np.quantile([float(x[key]) for x in rows], 0.25))


def p75(rows, key):
    return float(np.quantile([float(x[key]) for x in rows], 0.75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--flat-2048", type=Path, required=True)
    ap.add_argument("--ivf-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "flat_2048", "ivf_root", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    indexes = {}
    manifests = {}
    for _, nlist, _ in CONFIGS:
        if nlist in indexes:
            continue
        p = args.ivf_root / f"hints-b16000-nlist{nlist}-spherical.bin"
        indexes[nlist] = p
        manifests[str(nlist)] = json.loads(
            p.with_suffix(p.suffix + ".manifest.json").read_text()
        )

    methods = ["baseline-medoid", "flat-b2048"] + [name for name, _, _ in CONFIGS]
    config_by_name = {name: (nlist, nprobe) for name, nlist, nprobe in CONFIGS}

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
            # Nine methods and nine reps: every method occupies every ordinal
            # position exactly once, reducing systematic thermal/device bias.
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    kwargs = {}
                    if method == "flat-b2048":
                        kwargs["flat"] = args.flat_2048
                    elif method in config_by_name:
                        nlist, nprobe = config_by_name[method]
                        kwargs["ivf"] = indexes[nlist]
                        kwargs["nprobe"] = nprobe
                    row = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, **kwargs
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for method in methods:
        rr = [x for x in rows if x["method"] == method]
        payload = 0
        if method == "flat-b2048":
            payload = args.flat_2048.stat().st_size
        elif method in config_by_name:
            nlist, _ = config_by_name[method]
            payload = indexes[nlist].stat().st_size + nlist * (64 + 4)
        summary[method] = {
            "rounds": len(rr),
            "runtime_total_payload_bytes": int(payload),
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "qps_iqr": [p25(rr, "qps"), p75(rr, "qps")],
            "median_latency_us": med(rr, "mean_latency"),
            "latency_iqr_us": [p25(rr, "mean_latency"), p75(rr, "mean_latency")],
            "median_io_us": med(rr, "mean_io_time"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "cpu_iqr_us": [p25(rr, "mean_cpu_time"), p75(rr, "mean_cpu_time")],
            "median_pq_preprocess_us": med(rr, "mean_pq_preprocess_time"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
            "mean_hops": avg(rr, "mean_hops"),
        }

    routed = [name for name, _, _ in CONFIGS]
    pareto = []
    for m in routed:
        sm = summary[m]
        if not any(
            other != m
            and summary[other]["recall_percent"] >= sm["recall_percent"]
            and summary[other]["median_qps"] >= sm["median_qps"]
            and (
                summary[other]["recall_percent"] > sm["recall_percent"]
                or summary[other]["median_qps"] > sm["median_qps"]
            )
            for other in routed
        ):
            pareto.append(m)

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "implementation": "packed-direct coarse PQ + pooled vectorized fine scoring",
        "reps": args.reps,
        "methods": methods,
        "manifests": manifests,
        "summary": summary,
        "pareto_recall_qps": pareto,
    }
    save(args.out / "optimized-frontier-confirm.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
