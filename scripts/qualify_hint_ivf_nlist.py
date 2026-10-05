#!/usr/bin/env python3
"""Sweep ID-only hint-IVF bucket granularity and probe count."""
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
NLISTS = (64, 128, 256, 512)
NPROBES = (1, 2, 4, 8)


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


def run_one(binary, out, tag, queries, gt, index_prefix, *, flat_starts=None, hint_ivf=None, nprobe=None):
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
        raise ValueError(f"{tag}: expected one row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: recall missing")
    return row


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
    ap.add_argument("--flat-16000", type=Path, required=True)
    ap.add_argument("--ivf-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "flat_2048", "flat_16000", "ivf_dir", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    ivfs = {}
    manifests = {}
    for nlist in NLISTS:
        path = args.ivf_dir / f"hints-b16000-nlist{nlist}.bin"
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(path)
        ivfs[nlist] = path
        manifests[str(nlist)] = json.loads(manifest_path.read_text())

    methods = ["baseline-medoid", "flat-b2048", "flat-b16000"]
    methods += [f"ivf-n{n}-p{p}" for n in NLISTS for p in NPROBES]

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
                    elif method.startswith("ivf-"):
                        parts = method.split("-")
                        nlist = int(parts[1][1:])
                        nprobe = int(parts[2][1:])
                        kwargs["hint_ivf"] = ivfs[nlist]
                        kwargs["nprobe"] = nprobe
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
        if method == "baseline-medoid":
            state_bytes = 0
        elif method == "flat-b2048":
            state_bytes = args.flat_2048.stat().st_size
        elif method == "flat-b16000":
            state_bytes = args.flat_16000.stat().st_size
        else:
            nlist = int(method.split("-")[1][1:])
            state_bytes = ivfs[nlist].stat().st_size
        summary[method] = {
            "rounds": len(rr),
            "state_bytes": state_bytes,
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
            "mean_comparisons": avg(rr, "mean_comparisons"),
            "mean_hops": avg(rr, "mean_hops"),
        }

    frontier = []
    for nlist in NLISTS:
        for nprobe in NPROBES:
            name = f"ivf-n{nlist}-p{nprobe}"
            s = summary[name]
            frontier.append({
                "method": name,
                "nlist": nlist,
                "nprobe": nprobe,
                **s,
            })
    frontier.sort(key=lambda x: (-x["median_qps"], -x["recall_percent"]))

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "online_search": {"K": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "nlists": list(NLISTS),
        "nprobes": list(NPROBES),
        "manifests": manifests,
        "summary": summary,
        "ivf_points_by_qps": frontier,
    }
    save(args.out / "hint-ivf-nlist-results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
