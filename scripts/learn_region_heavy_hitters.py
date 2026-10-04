#!/usr/bin/env python3
"""Learn a fixed-budget navigation heavy-hitter cache for an arbitrary portal partition.

This is the area-count ablation learner. The router determines the cache key.
For each training query, every expanded vertex receives a skip-weighted score
equal to its first expansion position. The global top entries across
(region, vertex) pairs are selected under one fixed ID budget.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

ROUTER_MAGIC = b"GIPIP001"
CACHE_MAGIC = b"GIWPT001"


def fbin(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin byte size")
    return rows, dim, np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_router(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != ROUTER_MAGIC:
        raise ValueError("bad router header")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nfloat = nlist * dim
    off = 16
    centers = np.frombuffer(raw, dtype="<f4", count=nfloat, offset=off).reshape(nlist, dim).copy()
    off += nfloat * 4
    portal_ids = np.frombuffer(raw, dtype="<u4", count=nlist, offset=off).copy()
    off += nlist * 4
    portal_vecs = np.frombuffer(raw, dtype="<f4", count=nfloat, offset=off).reshape(nlist, dim).copy()
    off += nfloat * 4
    if off != len(raw):
        raise ValueError("router trailing bytes")
    return int(nlist), int(dim), centers, portal_ids, portal_vecs


def route_cells(q, centers, portal_vecs, nprobe):
    nlist = len(centers)
    k = min(int(nprobe), nlist)
    if k <= 0 or k > 32:
        raise ValueError("effective nprobe must be in 1..32")
    out = np.empty(len(q), dtype=np.int32)
    for start in range(0, len(q), 256):
        x = np.asarray(q[start:start+256], dtype=np.float32, order="C")
        coarse = x @ centers.T
        if k == nlist:
            top = np.broadcast_to(np.arange(nlist, dtype=np.int64), (len(x), nlist))
        else:
            top = np.argpartition(coarse, -k, axis=1)[:, -k:]
        for i in range(len(x)):
            cells = top[i]
            scores = portal_vecs[cells] @ x[i]
            out[start+i] = int(cells[int(np.argmax(scores))])
    return out


def write_cache(path: Path, nlist: int, selected):
    if len(selected) != nlist:
        raise ValueError("selected cell count mismatch")
    offsets = np.zeros(nlist + 1, dtype="<u4")
    flat = []
    for c, ids in enumerate(selected):
        seen = set()
        for raw in ids:
            v = int(raw)
            if v not in seen:
                seen.add(v)
                flat.append(v)
        offsets[c+1] = len(flat)
    ids = np.asarray(flat, dtype="<u4")
    with path.open("wb") as f:
        f.write(CACHE_MAGIC)
        f.write(struct.pack("<II", nlist, len(ids)))
        offsets.tofile(f)
        ids.tofile(f)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=2500)
    ap.add_argument("--nprobe", type=int, default=32)
    args = ap.parse_args()

    rows, qdim, q = fbin(args.queries)
    nlist, dim, centers, portal_ids, portal_vecs = load_router(args.router)
    if qdim != dim:
        raise ValueError(f"query/router dimension mismatch {qdim} != {dim}")

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query numbering mismatch")
    if len(records) > rows:
        raise ValueError("too many records")

    cells = route_cells(q[:len(records)], centers, portal_vecs, args.nprobe)
    score = defaultdict(int)
    support = defaultdict(int)

    first_matches = 0
    for qi, rec in enumerate(records):
        cell = int(cells[qi])
        ids = [int(v) for v in rec["ids"]]
        if ids and ids[0] == int(portal_ids[cell]):
            first_matches += 1
        first = {}
        for pos, vid in enumerate(ids):
            first.setdefault(vid, pos)
        for vid, pos in first.items():
            if pos <= 0:
                continue
            key = (cell, vid)
            score[key] += int(pos)
            support[key] += 1

    ranked = sorted(
        ((int(s), int(support[key]), int(key[0]), int(key[1])) for key, s in score.items() if s > 0),
        key=lambda x: (-x[0], -x[1], x[2], x[3]),
    )[:args.budget]

    selected = [[] for _ in range(nlist)]
    for _, _, cell, vid in ranked:
        selected[cell].append(vid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ids = write_cache(args.out, nlist, selected)
    counts = np.asarray([len(x) for x in selected], dtype=np.int64)
    demand = np.bincount(cells.astype(np.int64), minlength=nlist)

    manifest = {
        "training_queries": len(records),
        "router_nlist": nlist,
        "router_dim": dim,
        "nprobe_requested": args.nprobe,
        "nprobe_effective": min(args.nprobe, nlist),
        "budget_requested": args.budget,
        "cache_ids": int(len(ids)),
        "nonempty_regions": int(np.count_nonzero(counts)),
        "query_nonempty_regions": int(np.count_nonzero(demand)),
        "allocation": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "p95": float(np.quantile(counts, .95)),
            "max": int(counts.max()),
        },
        "demand": {
            "min": int(demand.min()),
            "median": float(np.median(demand)),
            "mean": float(demand.mean()),
            "p95": float(np.quantile(demand, .95)),
            "max": int(demand.max()),
        },
        "first_trace_vertex_matches_portal_fraction": first_matches / len(records),
        "score": "sum of first expansion position over training traversals",
        "cache_file": args.out.name,
        "cache_bytes": args.out.stat().st_size,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
