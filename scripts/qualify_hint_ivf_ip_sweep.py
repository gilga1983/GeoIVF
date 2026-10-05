#!/usr/bin/env python3
"""Compare inner-product hint-IVF granularities under one exact-recall campaign."""
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
NPROBES = (1, 2, 4, 8, 16)
NLISTS = (128, 256, 512)


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


def run_one(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    flat_starts: Path | None = None,
    hint_ivf: Path | None = None,
    nprobe: int | None = None,
):
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
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)

    if flat_starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(flat_starts)
    if hint_ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(hint_ivf)
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
        raise ValueError(f"{tag}: expected one result row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: exact recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def manifest_for(path: Path):
    return json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())


def estimated_pq_scores(manifest, nprobe):
    # Exact query-dependent bucket sizes vary. This is the balanced/mean-work
    # estimate used only to make the coarse/fine tradeoff interpretable.
    return float(manifest["nlist"] + nprobe * manifest["bucket_children_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--flat-2048", type=Path, required=True)
    ap.add_argument("--flat-16000", type=Path, required=True)
    ap.add_argument("--ip-root", type=Path, required=True)
    ap.add_argument("--spherical-256", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in (
        "binary", "queries", "gt", "index_prefix", "flat_2048", "flat_16000",
        "ip_root", "spherical_256", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    indexes = {}
    manifests = {}
    for nlist in NLISTS:
        p = args.ip_root / f"hints-b16000-nlist{nlist}-ip.bin"
        if not p.is_file():
            raise FileNotFoundError(p)
        indexes[f"ip{nlist}"] = p
        manifests[f"ip{nlist}"] = manifest_for(p)
    indexes["spherical256"] = args.spherical_256
    manifests["spherical256"] = manifest_for(args.spherical_256)

    methods = ["baseline-medoid", "flat-b2048", "flat-b16000"]
    methods += ["spherical256-p4", "spherical256-p8"]
    for nlist in NLISTS:
        methods += [f"ip{nlist}-p{p}" for p in NPROBES]

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
                    kwargs = {}
                    if method == "flat-b2048":
                        kwargs["flat_starts"] = args.flat_2048
                    elif method == "flat-b16000":
                        kwargs["flat_starts"] = args.flat_16000
                    elif "-p" in method:
                        family, raw_probe = method.rsplit("-p", 1)
                        kwargs["hint_ivf"] = indexes[family]
                        kwargs["nprobe"] = int(raw_probe)
                    row = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, **kwargs,
                    )
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    summary = {}
    for method in methods:
        rr = [x for x in rows if x["method"] == method]
        state_bytes = 0
        pq_est = None
        if method == "flat-b2048":
            state_bytes = args.flat_2048.stat().st_size
            pq_est = 2048
        elif method == "flat-b16000":
            state_bytes = args.flat_16000.stat().st_size
            pq_est = 16000
        elif "-p" in method:
            family, raw_probe = method.rsplit("-p", 1)
            p = int(raw_probe)
            state_bytes = indexes[family].stat().st_size
            pq_est = estimated_pq_scores(manifests[family], p)
        summary[method] = {
            "rounds": len(rr),
            "state_bytes": int(state_bytes),
            "estimated_router_pq_scores": pq_est,
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "mean_hops": avg(rr, "mean_hops"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
        }

    # Pick several useful optima rather than hiding the Pareto shape behind one score.
    routed = [m for m in methods if "-p" in m]
    best = {
        "highest_qps": max(routed, key=lambda m: summary[m]["median_qps"]),
        "lowest_io": min(routed, key=lambda m: summary[m]["mean_ios"]),
        "highest_recall": max(routed, key=lambda m: summary[m]["recall_percent"]),
        "best_qps_with_recall_ge_14": None,
    }
    feasible = [m for m in routed if summary[m]["recall_percent"] >= 14.0]
    if feasible:
        best["best_qps_with_recall_ge_14"] = max(
            feasible, key=lambda m: summary[m]["median_qps"]
        )

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "online_search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "partition_comparison": {
            "ip": "raw-inner-product medoids over actual hint vertices",
            "spherical_control": "cosine/spherical medoids from first hint-IVF experiment",
        },
        "nprobes": list(NPROBES),
        "manifests": manifests,
        "summary": summary,
        "selected_optima": best,
    }
    save(args.out / "hint-ivf-ip-sweep.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
