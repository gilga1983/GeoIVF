#!/usr/bin/env python3
"""Build exact top-k inner-product ground truth for .fbin query/base arrays.

The output matches the simple DiskANN benchmark ground-truth layout used by the
NavHints harness:
  uint32 nqueries, uint32 k,
  nqueries*k uint32 IDs,
  nqueries*k float32 distances.

For normalized Coveo vectors, maximizing inner product is equivalent to cosine.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_fbin_memmap(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"bad fbin size for {path}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim)), rows, dim


def stable_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k by score desc, ID asc on ties."""
    if scores.ndim != 2:
        raise ValueError("scores must be 2-D")
    n, m = scores.shape
    if k > m:
        raise ValueError("k exceeds base size")

    # Partial selection first, then exact stable ordering inside the candidate set.
    part = np.argpartition(scores, kth=m-k, axis=1)[:, m-k:]
    vals = np.take_along_axis(scores, part, axis=1)

    out_ids = np.empty((n, k), dtype=np.uint32)
    out_scores = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        ids = part[i].astype(np.int64, copy=False)
        v = vals[i]
        order = np.lexsort((ids, -v))
        ids = ids[order]
        out_ids[i] = ids.astype(np.uint32, copy=False)
        out_scores[i] = v[order].astype(np.float32, copy=False)
    return out_ids, out_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--query-batch", type=int, default=512)
    args = ap.parse_args()

    base, nb, db = load_fbin_memmap(args.base.resolve())
    queries, nq, dq = load_fbin_memmap(args.queries.resolve())
    if db != dq:
        raise ValueError(f"dimension mismatch {db} != {dq}")
    if not (0 <= args.start < nq):
        raise ValueError("invalid query start")
    count = nq - args.start if args.count is None else args.count
    if count <= 0 or args.start + count > nq:
        raise ValueError("invalid query count")
    if args.topk <= 0 or args.topk > nb:
        raise ValueError("invalid topk")
    if args.query_batch <= 0:
        raise ValueError("invalid query batch")

    ids = np.empty((count, args.topk), dtype=np.uint32)
    dists = np.empty((count, args.topk), dtype=np.float32)

    # Base is normally small enough for Coveo to materialize once; retaining the
    # memmap path keeps this generic and avoids duplicate disk formats.
    b = np.asarray(base, dtype=np.float32)
    for lo in range(0, count, args.query_batch):
        hi = min(count, lo + args.query_batch)
        q = np.asarray(queries[args.start + lo:args.start + hi], dtype=np.float32)
        scores = q @ b.T
        bi, bs = stable_topk(scores, args.topk)
        ids[lo:hi] = bi
        # DiskANN's inner-product benchmark convention is lower-is-better after
        # its internal transform, but recall uses IDs. Store negative similarity
        # as a distance-like quantity for diagnostics.
        dists[lo:hi] = -bs
        print(f"exact GT {hi}/{count}", flush=True)

    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(struct.pack("<II", count, args.topk))
        ids.astype("<u4", copy=False).tofile(f)
        dists.astype("<f4", copy=False).tofile(f)

    manifest = {
        "base_rows": nb,
        "dimension": db,
        "query_start": args.start,
        "query_count": count,
        "topk": args.topk,
        "metric": "maximum inner product; stored distance = negative inner product",
        "file": args.out.name,
        "bytes": args.out.stat().st_size,
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
