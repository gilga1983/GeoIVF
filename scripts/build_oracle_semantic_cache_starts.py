#!/usr/bin/env python3
"""Build oracle semantic-cache start files for the final 1K replay.

For each query q in [9000,10000), scan only previous requests in a rolling
cache drawn from q>=5000. Cache query embeddings are stored as fp16 for the
similarity scan. The nearest prior request contributes its exact successful
top-16 ground-truth IDs; the emitted start file contains the deployed NavHint
plus the first N distinct cached result IDs.

This is deliberately an upper-bound science screen. If it is promising, replace
oracle successful IDs with actual online DiskANN outputs.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

MAGIC_Q0 = 5000
EVAL_Q0 = 9000
EVAL_N = 1000


def fbin_memmap(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError("bad fbin")
    return np.memmap(path, mode="r", dtype="<f4", offset=8, shape=(rows, dim))


def load_gt(path: Path):
    raw = path.read_bytes()
    rows, k = struct.unpack("<II", raw[:8])
    if len(raw) != 8 + rows * k * 4:
        raise ValueError("bad gt")
    return np.frombuffer(raw, dtype="<u4", offset=8).reshape(rows, k)


def trace_starts(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != EVAL_N:
        raise ValueError(f"expected {EVAL_N} trace rows, got {len(rows)}")
    starts = np.empty(EVAL_N, dtype=np.uint32)
    for i, row in enumerate(rows):
        if int(row["query"]) != i or not row["ids"]:
            raise ValueError("bad trace row")
        starts[i] = int(row["ids"][0])
    return starts


def write_starts(path: Path, a: np.ndarray):
    a = np.asarray(a, dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        a.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt5000", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--capacities", default="128,256,512,1024,2048,4000")
    ap.add_argument("--result-counts", default="2,4,10")
    args = ap.parse_args()

    capacities = sorted({int(x) for x in args.capacities.split(",") if x.strip()})
    counts = sorted({int(x) for x in args.result_counts.split(",") if x.strip()})
    if min(capacities) <= 0 or min(counts) <= 0 or max(counts) > 15:
        raise ValueError("invalid sweep")

    q = fbin_memmap(args.queries.resolve())
    if q.shape != (10000, 768):
        raise ValueError(f"unexpected query shape {q.shape}")
    gt = load_gt(args.gt5000.resolve())
    if gt.shape[0] != 5000 or gt.shape[1] < 16:
        raise ValueError("unexpected heldout GT shape")

    nav = trace_starts(args.trace.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_starts(args.out_dir / "navhint-only.bin", nav[:, None])

    # Store the entire warm/eval suffix once in fp16, matching intended cache storage.
    q16 = np.asarray(q[MAGIC_Q0:MAGIC_Q0 + gt.shape[0]], dtype=np.float16)
    q32_eval = np.asarray(q[EVAL_Q0:EVAL_Q0 + EVAL_N], dtype=np.float32)

    result = {
        "policy": "rolling exact-scan semantic cache; fp16 cached queries; oracle prior exact result IDs",
        "warm_query_start": MAGIC_Q0,
        "eval_query_start": EVAL_Q0,
        "eval_queries": EVAL_N,
        "dimension": int(q.shape[1]),
        "capacities": {},
    }

    for cap in capacities:
        nearest = np.empty(EVAL_N, dtype=np.int32)
        similarities = np.empty(EVAL_N, dtype=np.float32)
        scan_sizes = np.empty(EVAL_N, dtype=np.int32)
        t0 = time.perf_counter()

        for ei in range(EVAL_N):
            absolute = EVAL_Q0 + ei
            rel = absolute - MAGIC_Q0
            lo = max(0, rel - cap)
            hi = rel
            cache = np.asarray(q16[lo:hi], dtype=np.float32)
            if len(cache) == 0:
                raise RuntimeError("empty semantic cache")
            sims = cache @ q32_eval[ei]
            local = int(np.argmax(sims))
            nearest[ei] = MAGIC_Q0 + lo + local
            similarities[ei] = float(sims[local])
            scan_sizes[ei] = len(cache)

        elapsed = time.perf_counter() - t0
        cmeta = {
            "capacity": cap,
            "query_vector_bytes": cap * 768 * 2,
            "result_id_bytes_top10": cap * 10 * 4,
            "total_payload_bytes_top10": cap * (768 * 2 + 10 * 4),
            "mean_scan_entries": float(scan_sizes.mean()),
            "scan_seconds_python_numpy": elapsed,
            "scan_us_per_query_python_numpy": 1e6 * elapsed / EVAL_N,
            "nearest_similarity": {
                "mean": float(similarities.mean()),
                "median": float(np.median(similarities)),
                "p05": float(np.quantile(similarities, 0.05)),
                "p95": float(np.quantile(similarities, 0.95)),
            },
            "arms": {},
        }

        for nres in counts:
            rows = np.empty((EVAL_N, 1 + nres), dtype=np.uint32)
            rows[:, 0] = nav
            duplicate_nav = 0
            for ei in range(EVAL_N):
                prior_abs = int(nearest[ei])
                prior_gt_row = prior_abs - MAGIC_Q0
                candidates = gt[prior_gt_row]
                chosen = []
                seen = {int(nav[ei])}
                for raw in candidates:
                    rid = int(raw)
                    if rid in seen:
                        duplicate_nav += 1
                        continue
                    seen.add(rid)
                    chosen.append(rid)
                    if len(chosen) == nres:
                        break
                if len(chosen) != nres:
                    raise RuntimeError("not enough distinct cached result IDs")
                rows[ei, 1:] = np.asarray(chosen, dtype=np.uint32)

            p = args.out_dir / f"semantic-c{cap}-r{nres}.bin"
            write_starts(p, rows)
            cmeta["arms"][str(nres)] = {
                "file": p.name,
                "start_width": 1 + nres,
                "duplicate_cached_ids_equal_navhint_skipped": duplicate_nav,
            }

        result["capacities"][str(cap)] = cmeta
        print(json.dumps({
            "capacity": cap,
            "scan_us_per_query_python_numpy": cmeta["scan_us_per_query_python_numpy"],
            "payload_mib_top10": cmeta["total_payload_bytes_top10"] / (1024 * 1024),
            "median_similarity": cmeta["nearest_similarity"]["median"],
        }), flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
