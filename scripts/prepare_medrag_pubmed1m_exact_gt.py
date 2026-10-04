#!/usr/bin/env python3
"""Exact inner-product ground truth for held-out MedRAG-Zipf on PubMed 1M.

Writes DiskANN truthset format:
  uint32 nqueries, uint32 k, then nqueries*k uint32 IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


def fbin(path: Path) -> np.memmap:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: byte-size mismatch")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--query-start", type=int, default=5000)
    ap.add_argument("--query-count", type=int, default=5000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    if args.k <= 0 or args.query_count <= 0 or args.threads <= 0:
        raise ValueError("k, query-count, and threads must be positive")

    import faiss

    base = fbin(args.base)
    queries = fbin(args.queries)
    if base.shape != (1_000_000, 768):
        raise ValueError(f"expected 1M x 768 base, got {base.shape}")
    if queries.shape[1] != 768:
        raise ValueError(f"expected 768-D queries, got {queries.shape}")
    if args.query_start < 0 or args.query_start + args.query_count > len(queries):
        raise ValueError("invalid query slice")

    q = np.asarray(
        queries[args.query_start:args.query_start + args.query_count],
        dtype=np.float32,
        order="C",
    )

    faiss.omp_set_num_threads(args.threads)
    index = faiss.IndexFlatIP(768)
    add_batch = 32768
    for start in range(0, len(base), add_batch):
        block = np.asarray(base[start:start + add_batch], dtype=np.float32, order="C")
        index.add(block)
        if (start // add_batch) % 4 == 0:
            print(f"added={min(start + len(block), len(base))}/{len(base)}", flush=True)
    if index.ntotal != len(base):
        raise AssertionError("FAISS base row count mismatch")

    # Search in modest batches so the temporary distance/result buffers stay small.
    all_ids = np.empty((len(q), args.k), dtype="<u4")
    all_dists = np.empty((len(q), args.k), dtype=np.float32)
    batch = 128
    for start in range(0, len(q), batch):
        qb = q[start:start + batch]
        dists, ids = index.search(qb, args.k)
        if np.any(ids < 0):
            raise RuntimeError("FAISS returned invalid exact neighbor ID")
        all_ids[start:start + len(qb)] = ids.astype("<u4", copy=False)
        all_dists[start:start + len(qb)] = dists
        print(f"searched={min(start + len(qb), len(q))}/{len(q)}", flush=True)

    # Spot-check exact top-1 with NumPy against the same full base.
    spot = [0, len(q)//2, len(q)-1]
    for qi in spot:
        best_id = -1
        best_score = -np.inf
        for start in range(0, len(base), 65536):
            block = np.asarray(base[start:start + 65536], dtype=np.float32, order="C")
            scores = block @ q[qi]
            j = int(np.argmax(scores))
            score = float(scores[j])
            if score > best_score:
                best_score = score
                best_id = start + j
        if best_id != int(all_ids[qi, 0]):
            raise AssertionError(
                f"exact spot check mismatch q={qi}: numpy={best_id}, faiss={int(all_ids[qi,0])}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", len(q), args.k))
        all_ids.tofile(f)
    expected = 8 + len(q) * args.k * 4
    if tmp.stat().st_size != expected:
        raise AssertionError("ground-truth byte-size mismatch")
    os.replace(tmp, args.out)

    meta = {
        "metric": "inner_product",
        "base_rows": int(len(base)),
        "dimension": 768,
        "query_source_rows": int(len(queries)),
        "query_start": args.query_start,
        "query_count": int(len(q)),
        "k": args.k,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "threads": args.threads,
        "base_sha256": sha256(args.base),
        "queries_sha256": sha256(args.queries),
        "gt_sha256": sha256(args.out),
        "first_top1_id": int(all_ids[0, 0]),
        "first_top1_ip": float(all_dists[0, 0]),
    }
    manifest = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
