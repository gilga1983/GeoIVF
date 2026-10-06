#!/usr/bin/env python3
"""Approximate ANN over DiskANN's PQ reconstructions, followed by exact validation.

This is intentionally a quality-feasibility experiment. It reconstructs each
DiskANN PQ code into its 768-D quantized vector, builds a FAISS HNSW over those
reconstructions, searches entirely in PQ-vector space, then validates the
returned IDs with the original full-precision PubMed vectors.

If this works, a deployment version can keep only a compact graph and use
DiskANN's existing PQ lookup tables instead of storing float reconstructions.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import faiss
import numpy as np

from analyze_pq_semantic_filter import fbin_memmap, gt_ids, load_pq


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def reconstruct_batch(pivots, offsets, codes, start, end):
    d = pivots.shape[1]
    out = np.empty((end - start, d), dtype=np.float32)
    cb = codes[start:end]
    for j in range(codes.shape[1]):
        a, b = int(offsets[j]), int(offsets[j + 1])
        out[:, a:b] = pivots[cb[:, j], a:b]
    return out


def build_or_load(index_path, pivots, offsets, codes, M, efc, add_batch):
    n, _ = codes.shape
    d = pivots.shape[1]
    if index_path.exists():
        started = time.monotonic()
        index = faiss.read_index(str(index_path))
        if index.d != d or index.ntotal != n:
            raise ValueError("cached HNSW shape mismatch")
        return index, 0.0, time.monotonic() - started, True

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc

    started = time.monotonic()
    for s in range(0, n, add_batch):
        e = min(n, s + add_batch)
        xb = reconstruct_batch(pivots, offsets, codes, s, e)
        index.add(xb)
        if s == 0 or e % 100000 < add_batch or e == n:
            print(f"hnsw_add={e}/{n}", flush=True)
    build_seconds = time.monotonic() - started
    faiss.write_index(index, str(index_path))
    return index, build_seconds, 0.0, False


def validate(base, queries, candidates, truth, k=10):
    nq, width = candidates.shape
    recalls = np.zeros(nq, dtype=np.float32)
    top_ids = np.empty((nq, k), dtype=np.int64)
    start = time.monotonic()
    for qi in range(nq):
        ids = candidates[qi].astype(np.int64, copy=False)
        ids = ids[ids >= 0]
        # HNSW should not duplicate IDs, but make validation robust.
        ids = np.unique(ids)
        xb = np.asarray(base[ids], dtype=np.float32)
        scores = xb @ queries[qi]
        take = min(k, len(ids))
        if take:
            local = np.argpartition(scores, -take)[-take:]
            local = local[np.argsort(scores[local])[::-1]]
            chosen = ids[local]
        else:
            chosen = np.empty(0, dtype=np.int64)
        top_ids[qi].fill(-1)
        top_ids[qi, : len(chosen)] = chosen
        tset = set(int(x) for x in truth[qi, :k])
        recalls[qi] = sum(int(x) in tset for x in chosen) / k
    seconds = time.monotonic() - start
    return recalls, top_ids, seconds


def raw_recall(candidates, truth, k=10):
    vals = []
    for qi in range(len(candidates)):
        c = set(int(x) for x in candidates[qi, :k] if x >= 0)
        t = set(int(x) for x in truth[qi, :k])
        vals.append(len(c & t) / k)
    return float(np.mean(vals))


def candidate_coverage(candidates, truth, k=10):
    vals = []
    full = 0
    for qi in range(len(candidates)):
        c = set(int(x) for x in candidates[qi] if x >= 0)
        t = set(int(x) for x in truth[qi, :k])
        hit = len(c & t)
        vals.append(hit / k)
        full += int(hit == k)
    return float(np.mean(vals)), full / len(candidates)


def main():
    ap = argparse.ArgumentParser()
    for name in ("base", "queries", "gt5000", "pq-pivots", "pq-codes", "index-cache", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--query-offset", type=int, default=9000)
    ap.add_argument("--gt-offset", type=int, default=4000)
    ap.add_argument("--query-count", type=int, default=1000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=80)
    ap.add_argument("--add-batch", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--candidate-budgets", default="10,20,50,100,200")
    ap.add_argument("--ef-search", default="32,64,128,256")
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads)

    base = fbin_memmap(args.base.resolve())
    qall = fbin_memmap(args.queries.resolve())
    queries = np.ascontiguousarray(
        qall[args.query_offset: args.query_offset + args.query_count],
        dtype=np.float32,
    )
    gtall = gt_ids(args.gt5000.resolve())
    truth = gtall[args.gt_offset: args.gt_offset + args.query_count]
    if len(queries) != args.query_count or len(truth) != args.query_count:
        raise ValueError("held-out range mismatch")

    pivots, offsets, codes = load_pq(args.pq_pivots.resolve(), args.pq_codes.resolve())
    if codes.shape[0] != base.shape[0] or pivots.shape[1] != base.shape[1]:
        raise ValueError("PQ/base shape mismatch")

    budgets = sorted({int(x) for x in args.candidate_budgets.split(",") if x.strip()})
    efs = sorted({int(x) for x in args.ef_search.split(",") if x.strip()})
    if not budgets or min(budgets) < 10:
        raise ValueError("candidate budgets must be >=10")
    if max(budgets) > max(efs):
        # HNSW permits k > efSearch in FAISS, but that is not a sensible screen.
        efs.append(max(budgets))
        efs = sorted(set(efs))

    index, build_s, load_s, cached = build_or_load(
        args.index_cache.resolve(), pivots, offsets, codes,
        args.M, args.ef_construction, args.add_batch,
    )

    result = {
        "experiment": "FAISS HNSW over exact DiskANN PQ reconstructions, exact full-vector validation",
        "workload": {
            "database_vectors": int(base.shape[0]),
            "dimension": int(base.shape[1]),
            "heldout_queries": int(len(queries)),
            "query_offset": args.query_offset,
            "truth_k_available": int(truth.shape[1]),
        },
        "pq": {
            "chunks": int(codes.shape[1]),
            "centers": int(pivots.shape[0]),
            "code_bytes_per_vector": int(codes.shape[1]),
        },
        "hnsw": {
            "M": args.M,
            "efConstruction": args.ef_construction,
            "threads": args.threads,
            "cached_index": cached,
            "build_seconds": build_s,
            "load_seconds": load_s,
            "serialized_index_bytes": int(args.index_cache.stat().st_size),
        },
        "results": [],
    }

    for ef in efs:
        index.hnsw.efSearch = ef
        for budget in budgets:
            if budget > ef:
                continue
            started = time.monotonic()
            pq_scores, candidates = index.search(queries, budget)
            search_s = time.monotonic() - started

            coverage, full_coverage = candidate_coverage(candidates, truth, 10)
            recalls, _, validation_s = validate(base, queries, candidates, truth, 10)
            raw10 = raw_recall(candidates, truth, 10)

            row = {
                "efSearch": ef,
                "candidate_budget": budget,
                "pq_ann_raw_top10_recall": raw10,
                "candidate_true_top10_coverage": coverage,
                "queries_with_all_true_top10_in_candidates_fraction": full_coverage,
                "exact_validated_recall_at_10": float(recalls.mean()),
                "recall_p05": float(np.quantile(recalls, .05)),
                "recall_median": float(np.median(recalls)),
                "search_ms_per_query": 1000.0 * search_s / len(queries),
                "exact_validation_ms_per_query": 1000.0 * validation_s / len(queries),
                "total_screen_plus_validation_ms_per_query": 1000.0 * (search_s + validation_s) / len(queries),
            }
            result["results"].append(row)
            print(json.dumps(row), flush=True)

    # Pareto-like concise ranking: highest recall, then lowest screen time.
    result["ranking"] = sorted(
        result["results"],
        key=lambda r: (-r["exact_validated_recall_at_10"], r["search_ms_per_query"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
