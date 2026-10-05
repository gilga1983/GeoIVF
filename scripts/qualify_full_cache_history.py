#!/usr/bin/env python3
"""Compare frozen, causal-warm, and oracle full-history NavHints states.

History/state protocol:
  block 5000:5999 uses h5000 before those queries are folded in;
  block 6000:6999 uses h6000;
  ...
  block 9000:9999 uses h9000.

The frozen control always uses h5000. The oracle arm always uses h10000 and is
explicitly leaky: it quantifies headroom only.

Runs both the L=1 Recall@1 diagnostic and a realistic K=L=10 point.
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
BLOCK = 1000
BLOCK_STARTS = (5000, 6000, 7000, 8000, 9000)
SEARCHES = (("r1-l1", 1, 1), ("r10-l10", 10, 10))
START_MAGIC = b"GIDST001"


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


def slice_fbin(src: Path, dst: Path, start: int, count: int):
    rows, dim = fbin_shape(src)
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError("bad fbin slice")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", count, dim))
        remaining = count * dim * 4
        while remaining:
            chunk = fin.read(min(16 << 20, remaining))
            if not chunk:
                raise ValueError("truncated fbin")
            fout.write(chunk)
            remaining -= len(chunk)


def slice_truthset(src: Path, dst: Path, start: int, count: int):
    raw = src.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated truthset")
    npts, dim = struct.unpack("<II", raw[:8])
    ids_bytes = npts * dim * 4
    ids_only = 8 + ids_bytes
    ids_and_dists = 8 + 2 * ids_bytes
    if len(raw) not in (ids_only, ids_and_dists):
        raise ValueError("unsupported truthset size")
    if start < 0 or count <= 0 or start + count > npts:
        raise ValueError("bad truthset slice")

    row_bytes = dim * 4
    ids_base = 8
    ids = raw[ids_base + start * row_bytes : ids_base + (start + count) * row_bytes]
    with dst.open("wb") as f:
        f.write(struct.pack("<II", count, dim))
        f.write(ids)
        if len(raw) == ids_and_dists:
            dist_base = 8 + ids_bytes
            dists = raw[
                dist_base + start * row_bytes :
                dist_base + (start + count) * row_bytes
            ]
            f.write(dists)


def load_ids(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError(f"bad landmark file {path}")
    n, reserved = struct.unpack("<II", raw[8:16])
    if reserved != 0 or len(raw) != 16 + n * 4:
        raise ValueError(f"bad landmark file size {path}")
    return np.frombuffer(raw, dtype="<u4", count=n, offset=16).copy()


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            found.append(obj)
        else:
            for value in obj.values():
                found.extend(result_rows(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(result_rows(value))
    return found


def run_one(binary, out, tag, queries, gt, index_prefix, ivf, recall_at, lval):
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
                    "search_list": [lval],
                    "beam_width": IO_BEAM,
                    "recall_at": recall_at,
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
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

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
        raise ValueError(f"{tag}: expected one result row, got {len(rows)}")
    row = dict(rows[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: exact recall missing")
    return row


def avg(rows, key):
    return float(np.mean([float(x[key]) for x in rows]))


def median(rows, key):
    return float(np.median([float(x[key]) for x in rows]))


def aggregate_qps(rows, reps):
    by_rep = {}
    for row in rows:
        by_rep.setdefault(int(row["rep"]), []).append(row)
    values = []
    for rep in range(reps):
        rr = by_rep[rep]
        if len(rr) != len(BLOCK_STARTS):
            raise ValueError("missing block in aggregate QPS")
        seconds = sum(BLOCK / float(x["qps"]) for x in rr)
        values.append((BLOCK * len(BLOCK_STARTS)) / seconds)
    return float(np.median(values)), values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--heldout-gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--state-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    for name in ("binary", "queries", "heldout_gt", "index_prefix", "state_root", "work", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")

    histories = (5000, 6000, 7000, 8000, 9000, 10000)
    ivfs = {}
    landmarks = {}
    for h in histories:
        root = args.state_root / f"h{h}"
        ivf = root / "hints-b16000-nlist512-spherical.bin"
        lm = root / "landmarks-b16000.bin"
        if not ivf.is_file() or not lm.is_file():
            raise FileNotFoundError(f"missing h{h} state")
        ivfs[h] = ivf
        landmarks[h] = load_ids(lm)

    overlap = {}
    base_set = set(map(int, landmarks[5000]))
    for h in histories:
        cur = set(map(int, landmarks[h]))
        inter = len(base_set & cur)
        overlap[str(h)] = {
            "intersection_with_h5000": inter,
            "fraction_of_16k_retained": inter / 16000.0,
            "jaccard_with_h5000": inter / len(base_set | cur),
        }

    blocks = {}
    for start in BLOCK_STARTS:
        q = args.work / f"q{start}-{start+BLOCK}.fbin"
        gt = args.work / f"gt{start}-{start+BLOCK}.bin"
        slice_fbin(args.queries, q, start, BLOCK)
        # heldout GT row zero corresponds to workload query 5000.
        slice_truthset(args.heldout_gt, gt, start - 5000, BLOCK)
        blocks[start] = (q, gt)

    methods = ("frozen", "causal", "oracle")
    rows = []
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))

    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Rotate method order per rep and block to reduce timing bias.
            for search_name, recall_at, lval in SEARCHES:
                for rep in range(args.reps):
                    for bi, start in enumerate(BLOCK_STARTS):
                        history = start
                        order = list(methods)
                        shift = (rep + bi) % len(order)
                        order = order[shift:] + order[:shift]
                        for method in order:
                            if method == "frozen":
                                h = 5000
                            elif method == "causal":
                                h = history
                            else:
                                h = 10000
                            q, gt = blocks[start]
                            tag = f"{search_name}-r{rep}-b{start}-{method}-h{h}"
                            row = run_one(
                                args.binary,
                                args.out,
                                tag,
                                q,
                                gt,
                                args.index_prefix,
                                ivfs[h],
                                recall_at,
                                lval,
                            )
                            rows.append({
                                "search": search_name,
                                "recall_at": recall_at,
                                "L": lval,
                                "rep": rep,
                                "block_start": start,
                                "method": method,
                                "history_rows": h,
                                **row,
                            })
                            save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)

    summary = {}
    blockwise = {}
    for search_name, _, _ in SEARCHES:
        summary[search_name] = {}
        blockwise[search_name] = {}
        sr = [x for x in rows if x["search"] == search_name]
        for method in methods:
            rr = [x for x in sr if x["method"] == method]
            qps, qps_reps = aggregate_qps(rr, args.reps)
            summary[search_name][method] = {
                "recall_percent": avg(rr, "recall"),
                "mean_ios": avg(rr, "mean_ios"),
                "median_aggregate_qps": qps,
                "aggregate_qps_by_rep": qps_reps,
                "median_cpu_us": median(rr, "mean_cpu_time"),
                "median_latency_us": median(rr, "mean_latency"),
            }

        for start in BLOCK_STARTS:
            blockwise[search_name][str(start)] = {}
            for method in methods:
                rr = [
                    x for x in sr
                    if x["method"] == method and x["block_start"] == start
                ]
                blockwise[search_name][str(start)][method] = {
                    "history_rows": int(rr[0]["history_rows"]),
                    "recall_percent": avg(rr, "recall"),
                    "mean_ios": avg(rr, "mean_ios"),
                    "median_qps": median(rr, "qps"),
                }

    deltas = {}
    for search_name, _, _ in SEARCHES:
        s = summary[search_name]
        deltas[search_name] = {
            "causal_minus_frozen_recall_points":
                s["causal"]["recall_percent"] - s["frozen"]["recall_percent"],
            "causal_minus_frozen_ios":
                s["causal"]["mean_ios"] - s["frozen"]["mean_ios"],
            "causal_over_frozen_qps":
                s["causal"]["median_aggregate_qps"] / s["frozen"]["median_aggregate_qps"],
            "oracle_minus_frozen_recall_points":
                s["oracle"]["recall_percent"] - s["frozen"]["recall_percent"],
            "oracle_minus_causal_recall_points":
                s["oracle"]["recall_percent"] - s["causal"]["recall_percent"],
            "oracle_minus_frozen_ios":
                s["oracle"]["mean_ios"] - s["frozen"]["mean_ios"],
        }

    result = {
        "protocol": {
            "frozen": "all heldout blocks use vocabulary learned from queries 0:5000",
            "causal": "block [t,t+1000) uses only traces [0,t)",
            "oracle": "all heldout blocks use traces 0:10000; intentionally leaks test queries",
            "hints": 16000,
            "nlist": 512,
            "nprobe": 8,
            "implementation": "packed-direct coarse PQ + pooled vectorized fine scoring",
        },
        "vocabulary_overlap_with_h5000": overlap,
        "summary": summary,
        "blockwise": blockwise,
        "deltas": deltas,
    }
    save(args.out / "full-cache-history.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
