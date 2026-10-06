#!/usr/bin/env python3
"""Evaluate one-stage vs equal-memory two-stage NavHints."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

THREADS = 4
BEAM = 8
K = 10
LS = (10, 20, 40, 80, 160)
RUN_TIMEOUT_SECONDS = 600
HEARTBEAT_SECONDS = 30


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
            for value in obj.values():
                out.extend(result_rows(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(result_rows(value))
    return out


def run_one(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    start_ivf: Path,
    start_probe: int,
    stage_ivf: Path | None = None,
    stage_probe: int | None = None,
    stage_hops: int | str | None = None,
    progressive_hints: bool = False,
):
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
    logp = out / f"{tag}.log"
    save(inp, cfg)

    env = os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL",
        "DISKANN_STATIC_CACHE_IDS_FILE",
        "DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_STAGE_HINT_IVF_FILE",
        "DISKANN_STAGE_HINT_IVF_NPROBE",
        "DISKANN_STAGE_HINT_HOPS",
        "DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)

    env["DISKANN_HINT_IVF_FILE"] = str(start_ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = str(start_probe)
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    if progressive_hints:
        env["DISKANN_PROGRESSIVE_HINTS"] = "1"
    if stage_ivf is not None:
        if stage_probe is None or stage_hops is None:
            raise ValueError("stage probe/hops required")
        env["DISKANN_STAGE_HINT_IVF_FILE"] = str(stage_ivf)
        env["DISKANN_STAGE_HINT_IVF_NPROBE"] = str(stage_probe)
        env["DISKANN_STAGE_HINT_HOPS"] = str(stage_hops)

    cmd = [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)]
    started = time.monotonic()
    print(f"starting {tag}", flush=True)
    with logp.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                rc = proc.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(
                    f"benchmark-heartbeat tag={tag} pid={proc.pid} elapsed={elapsed:.0f}s "
                    f"log_bytes={logp.stat().st_size} load1={os.getloadavg()[0]:.2f}",
                    flush=True,
                )
                if elapsed >= RUN_TIMEOUT_SECONDS:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise TimeoutError(f"{tag} exceeded {RUN_TIMEOUT_SECONDS}s")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    rows = sorted(result_rows(json.loads(output.read_text())), key=lambda r: int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS):
        raise ValueError(f"{tag}: bad L grid")
    print(f"completed {tag} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return rows


def aggregate(reps):
    out = {}
    for l in LS:
        rows = [r for rr in reps for r in rr if int(r["search_l"]) == l]
        out[str(l)] = {
            "rounds": len(rows),
            "recall_percent": float(np.mean([float(r["recall"]) for r in rows])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in rows])),
            "median_qps": float(np.median([float(r["qps"]) for r in rows])),
            "median_latency_us": float(np.median([float(r["mean_latency"]) for r in rows])),
            "median_cpu_us": float(np.median([float(r["mean_cpu_time"]) for r in rows])),
            "mean_comparisons": float(np.mean([float(r["mean_comparisons"]) for r in rows])),
            "mean_hops": float(np.mean([float(r["mean_hops"]) for r in rows])),
        }
    return out


def monotone(summary):
    pts, best = [], -1e99
    for l in LS:
        row = summary[str(l)]
        r = float(row["recall_percent"])
        if r + 1e-9 >= best:
            pts.append((r, row))
            best = max(best, r)
    return pts


def interp(summary, target, field):
    pts = monotone(summary)
    if target < pts[0][0] or target > pts[-1][0]:
        return None
    if target == pts[0][0]:
        return float(pts[0][1][field])
    for (lo_r, lo), (hi_r, hi) in zip(pts, pts[1:]):
        if lo_r <= target <= hi_r:
            if hi_r <= lo_r + 1e-12:
                return float(hi[field])
            a = (target - lo_r) / (hi_r - lo_r)
            return float(lo[field]) + a * (float(hi[field]) - float(lo[field]))
    return None


def payload(path: Path):
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    return path.stat().st_size + int(manifest["nlist"]) * (64 + 4)


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "queries", "gt", "index-prefix", "ivf-16k", "ivf-start8k", "ivf-stage8k", "work", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "gt", "index_prefix", "ivf_16k", "ivf_start8k", "ivf_stage8k", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    methods = ("start16k", "start8k", "staged8k8")
    runs = {m: [] for m in methods}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            print(f"waiting for speed-device lock: {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("acquired speed-device lock", flush=True)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                print(f"rep {rep}: {' '.join(order)}", flush=True)
                for method in order:
                    if method == "start16k":
                        kw = dict(start_ivf=args.ivf_16k, start_probe=8)
                    elif method == "start8k":
                        kw = dict(start_ivf=args.ivf_start8k, start_probe=4)
                    else:
                        kw = dict(
                            start_ivf=args.ivf_start8k,
                            start_probe=4,
                            stage_ivf=args.ivf_stage8k,
                            stage_probe=4,
                            stage_hops=8,
                        )
                    rr = run_one(
                        args.binary, args.out, f"r{rep}-{method}",
                        held, args.gt, args.index_prefix, **kw
                    )
                    runs[method].append(rr)
                    save(args.out / "runs.partial.json", runs)
                    print(
                        f"finished rep={rep} method={method} "
                        f"L10 recall={float(rr[0]['recall']):.3f} ios={float(rr[0]['mean_ios']):.3f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {m: aggregate(runs[m]) for m in methods}
    memory = {
        "start16k": payload(args.ivf_16k),
        "start8k": payload(args.ivf_start8k),
        "staged8k8": payload(args.ivf_start8k) + payload(args.ivf_stage8k),
    }

    fixed = {}
    lo = max(summary["start16k"]["10"]["recall_percent"], summary["staged8k8"]["10"]["recall_percent"])
    hi = min(summary["start16k"]["160"]["recall_percent"], summary["staged8k8"]["160"]["recall_percent"])
    for target in (35.0, 45.0, 55.0, 65.0):
        if not (lo <= target <= hi):
            continue
        item = {}
        for method in methods:
            io = interp(summary[method], target, "mean_ios")
            lat = interp(summary[method], target, "median_latency_us")
            item[method] = {
                "mean_ios": io,
                "latency_us": lat,
                "runtime_payload_bytes": memory[method],
            }
        base = item["start16k"]
        stage = item["staged8k8"]
        item["staged_vs_start16k"] = {
            "io_change_percent": 100.0 * (stage["mean_ios"] / base["mean_ios"] - 1.0),
            "latency_change_percent": 100.0 * (stage["latency_us"] / base["latency_us"] - 1.0),
        }
        fixed[str(target)] = item

    same_l = {}
    for l in LS:
        base = summary["start16k"][str(l)]
        stage = summary["staged8k8"][str(l)]
        same_l[str(l)] = {
            "start16k_recall": base["recall_percent"],
            "staged_recall": stage["recall_percent"],
            "recall_delta_points": stage["recall_percent"] - base["recall_percent"],
            "start16k_ios": base["mean_ios"],
            "staged_ios": stage["mean_ios"],
            "io_delta_percent": 100.0 * (stage["mean_ios"] / base["mean_ios"] - 1.0),
            "start16k_latency_us": base["median_latency_us"],
            "staged_latency_us": stage["median_latency_us"],
            "latency_delta_percent": 100.0 * (stage["median_latency_us"] / base["median_latency_us"] - 1.0),
        }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "training": {
            "source": "ordinary medoid-start DiskANN teacher L=4, first 5000 queries",
            "start_score": "sum p_q(v)",
            "stage8_score": "sum max(p_q(v)-8,0)",
        },
        "arms": {
            "start16k": "16K hints, 512 cells, probe 8, start only",
            "start8k": "8K hints, 256 cells, probe 4, start only",
            "staged8k8": "8K start + independent 8K residual-stage-8 hints, each 256 cells/probe 4",
        },
        "memory_payload_bytes": memory,
        "memory_delta_staged_minus_start16k_bytes": memory["staged8k8"] - memory["start16k"],
        "evaluation": {"K": K, "Ls": list(LS), "beam": BEAM, "threads": THREADS, "repetitions": args.reps},
        "summary": summary,
        "same_L_comparison": same_l,
        "fixed_recall_comparison": fixed,
    }
    save(args.out / "staged-navhints.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
