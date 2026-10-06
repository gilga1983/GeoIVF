#!/usr/bin/env python3
"""Compare query-key and value-ID semantic cache lookup using actual prior DiskANN outputs.

The rolling cache contains one entry per previous request. Each entry's actual
DiskANN top-10 output is available as page-resident sibling metadata.

Two lookup policies differ only in the cache key:
  query_pq: scan a 64-byte PQ code of the previous query.
  value_id: scan the existing DiskANN PQ code of that request's best returned ID.

Both select one previous request, emit only its anchor at search start, and
expose the remaining nine result IDs only after that anchor page expands.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from analyze_pq_semantic_filter import fbin_memmap, load_pq, lut_for_query
from compare_semantic_cache_query_keys import encode_queries_pq, score_pq_cache

WARM0 = 5000
EVAL0 = 9000
N = 1000


def read_ids(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rows, k = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows * k)
    if a.size != rows * k:
        raise ValueError("truncated result dump")
    return a.reshape(rows, k)


def write_rows(path: Path, a: np.ndarray) -> None:
    a = np.asarray(a, dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        a.tofile(f)


def unique_results(row: np.ndarray) -> list[int]:
    seen = set()
    out = []
    for raw in row:
        x = int(raw)
        if x not in seen:
            seen.add(x)
            out.append(x)
    if len(out) < 10:
        raise ValueError("producer row contains duplicate/insufficient results")
    return out[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    for n in ("queries", "pq-pivots", "pq-codes", "results", "out-dir"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--capacities", default="512,2048")
    args = ap.parse_args()

    caps = sorted({int(x) for x in args.capacities.split(",") if x.strip()})
    queries = fbin_memmap(args.queries.resolve())
    pivots, offsets, db_codes = load_pq(args.pq_pivots.resolve(), args.pq_codes.resolve())
    results = read_ids(args.results.resolve())
    if queries.shape != (10000, 768):
        raise ValueError(f"unexpected query shape {queries.shape}")
    if results.shape != (5000, 10):
        raise ValueError(f"unexpected result shape {results.shape}")

    q_warm = np.asarray(queries[WARM0:], dtype=np.float32)
    q_eval = np.asarray(queries[EVAL0:EVAL0 + N], dtype=np.float32)
    print("encoding previous queries with DiskANN PQ codebook", flush=True)
    q_codes = encode_queries_pq(q_warm, pivots, offsets)

    anchors = results[:, 0].astype(np.int64)
    if np.any(anchors < 0) or np.any(anchors >= db_codes.shape[0]):
        raise ValueError("anchor outside database PQ table")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy": "actual prior DiskANN results; compare previous-query PQ key versus best-result value ID",
        "producer": str(args.results),
        "capacities": {},
    }

    for cap in caps:
        query_match = np.empty(N, dtype=np.int32)
        value_match = np.empty(N, dtype=np.int32)
        tq = 0.0
        tv = 0.0
        duplicate_anchor_fracs = []

        for ei, q in enumerate(q_eval):
            rel = (EVAL0 + ei) - WARM0
            lo = max(0, rel - cap)
            hi = rel
            if lo >= hi:
                raise RuntimeError("empty rolling cache")

            t0 = time.perf_counter()
            qs = score_pq_cache(q, q_codes[lo:hi], pivots, offsets)
            query_match[ei] = lo + int(np.argmax(qs))
            tq += time.perf_counter() - t0

            candidate_anchors = anchors[lo:hi]
            t0 = time.perf_counter()
            # Score cached values using the database PQ codes already resident in DiskANN.
            lut = lut_for_query(q, pivots, offsets)
            cb = db_codes[candidate_anchors]
            vs = np.zeros(len(candidate_anchors), dtype=np.float32)
            for j in range(db_codes.shape[1]):
                vs += lut[j, cb[:, j]]
            # np.argmax is stable: ties choose the older duplicate. Tie policy is
            # intentionally left simple for this first isolated experiment.
            value_match[ei] = lo + int(np.argmax(vs))
            tv += time.perf_counter() - t0

            duplicate_anchor_fracs.append(
                1.0 - len(np.unique(candidate_anchors)) / len(candidate_anchors)
            )

        agree = float(np.mean(query_match == value_match))
        cmeta = {
            "capacity": cap,
            "query_key_directory_bytes": cap * (q_codes.shape[1] + 4),
            "value_id_directory_bytes": cap * 4,
            "query_key_python_scan_us_per_query": 1e6 * tq / N,
            "value_id_python_scan_us_per_query": 1e6 * tv / N,
            "selected_request_agreement_fraction": agree,
            "mean_duplicate_anchor_fraction_in_window": float(np.mean(duplicate_anchor_fracs)),
            "arms": {},
        }

        for name, matches in (("query_pq", query_match), ("value_id", value_match)):
            rows = np.empty((N, 10), dtype=np.uint32)
            selected_anchor_same_as_other = 0
            for ei, ri in enumerate(matches):
                ids = unique_results(results[int(ri)])
                rows[ei] = np.asarray(ids, dtype=np.uint32)
            p = args.out_dir / f"{name}-c{cap}.bin"
            write_rows(p, rows)
            cmeta["arms"][name] = {"file": p.name}

        same_anchor = np.mean(
            results[query_match, 0].astype(np.uint32)
            == results[value_match, 0].astype(np.uint32)
        )
        cmeta["selected_anchor_agreement_fraction"] = float(same_anchor)
        manifest["capacities"][str(cap)] = cmeta
        print(json.dumps({
            "capacity": cap,
            "query_dir_kib": cmeta["query_key_directory_bytes"] / 1024,
            "value_dir_kib": cmeta["value_id_directory_bytes"] / 1024,
            "request_agreement": agree,
            "anchor_agreement": float(same_anchor),
            "duplicate_anchor_fraction": cmeta["mean_duplicate_anchor_fraction_in_window"],
            "query_scan_us": cmeta["query_key_python_scan_us_per_query"],
            "value_scan_us": cmeta["value_id_python_scan_us_per_query"],
        }), flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
