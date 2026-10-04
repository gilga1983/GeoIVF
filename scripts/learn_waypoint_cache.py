#!/usr/bin/env python3
"""Learn demand-adaptive waypoint caches from portal-seeded DiskANN traces.

Each query is routed to the same static IP-portal cell used at search time.
For a query trace [v0, v1, ...], candidate vi receives benefit i: the number
of graph-expansion I/Os before it. A selected cell/vertex pair covers a query
by the maximum benefit among selected candidates on that query's trace.

Selection is global under a fixed ID budget and uses lazy greedy marginal gain,
so hot cells can receive more entries while redundant waypoints get diminishing
credit. This is a facility-location/coverage proxy for I/O saved.
"""
from __future__ import annotations

import argparse
import heapq
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

ROUTER_MAGIC = b"GIPIP001"
CACHE_MAGIC = b"GIWPT001"
DIM = 768
NLIST = 1024


def fbin(path: Path) -> np.memmap:
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if dim != DIM:
        raise ValueError(f"expected dim {DIM}, got {dim}")
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin byte size")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_router(path: Path):
    raw = path.read_bytes()
    if raw[:8] != ROUTER_MAGIC:
        raise ValueError("wrong portal-router magic")
    nlist, dim = struct.unpack("<II", raw[8:16])
    if (nlist, dim) != (NLIST, DIM):
        raise ValueError(f"unexpected router shape {nlist}x{dim}")
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
    return centers, portal_ids, portal_vecs


def route_cells(q: np.ndarray, centers: np.ndarray, portal_vecs: np.ndarray, nprobe: int):
    if not 1 <= nprobe <= 32:
        raise ValueError("nprobe must be in 1..32")
    n = len(q)
    out = np.empty(n, dtype=np.int32)
    batch = 256
    for start in range(0, n, batch):
        x = np.asarray(q[start:start+batch], dtype=np.float32, order="C")
        coarse = x @ centers.T
        # Unordered top-nprobe is fine; winner is selected by portal IP below.
        top = np.argpartition(coarse, -nprobe, axis=1)[:, -nprobe:]
        for i in range(len(x)):
            cells = top[i]
            ps = portal_vecs[cells] @ x[i]
            out[start+i] = int(cells[int(np.argmax(ps))])
        print(f"routed={min(start+len(x), n)}/{n}", flush=True)
    return out


def write_cache(path: Path, selected_by_cell: list[list[int]]):
    if len(selected_by_cell) != NLIST:
        raise ValueError("wrong cell count")
    offsets = np.zeros(NLIST + 1, dtype="<u4")
    flat = []
    for c, ids in enumerate(selected_by_cell):
        # Stable unique list, preserving greedy selection order within the cell.
        seen = set()
        uniq = []
        for v in ids:
            v = int(v)
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        flat.extend(uniq)
        offsets[c+1] = len(flat)
    ids = np.asarray(flat, dtype="<u4")
    with path.open("wb") as f:
        f.write(CACHE_MAGIC)
        f.write(struct.pack("<II", NLIST, len(ids)))
        offsets.tofile(f)
        ids.tofile(f)
    expected = 16 + (NLIST + 1) * 4 + len(ids) * 4
    if path.stat().st_size != expected:
        raise AssertionError("waypoint cache byte-size mismatch")
    return offsets, ids


def learn(records, cells, budget: int, min_support: int):
    # candidate -> [(query index, benefit)]
    occ = defaultdict(list)
    query_total_prefix = np.zeros(len(records), dtype=np.int32)
    candidate_support = defaultdict(int)

    for qi, rec in enumerate(records):
        ids = [int(x) for x in rec["ids"]]
        if not ids:
            continue
        query_total_prefix[qi] = max(0, len(ids) - 1)
        cell = int(cells[qi])
        # Search should not expand the same graph node twice, but use first
        # occurrence defensively.
        first = {}
        for pos, vid in enumerate(ids):
            first.setdefault(vid, pos)
        for vid, pos in first.items():
            benefit = int(pos)
            if benefit <= 0:
                continue
            key = (cell, int(vid))
            occ[key].append((qi, benefit))
            candidate_support[key] += 1

    eligible = {
        k: v for k, v in occ.items()
        if candidate_support[k] >= min_support
    }

    current = np.zeros(len(records), dtype=np.int32)
    heap = []
    for key, lst in eligible.items():
        gain = sum(benefit for _, benefit in lst)
        if gain > 0:
            heapq.heappush(heap, (-gain, key))

    selected = []
    selected_by_cell = [[] for _ in range(NLIST)]
    marginal_gains = []

    while heap and len(selected) < budget:
        neg_bound, key = heapq.heappop(heap)
        lst = eligible[key]
        gain = 0
        for qi, benefit in lst:
            if benefit > current[qi]:
                gain += benefit - int(current[qi])
        if gain <= 0:
            continue

        # Lazy greedy: if recomputed gain no longer dominates the next stale
        # upper bound, reinsert with its exact current gain.
        next_bound = -heap[0][0] if heap else -1
        if gain < next_bound:
            heapq.heappush(heap, (-gain, key))
            continue

        selected.append(key)
        marginal_gains.append(gain)
        cell, vid = key
        selected_by_cell[cell].append(vid)
        for qi, benefit in lst:
            if benefit > current[qi]:
                current[qi] = benefit

    counts = np.asarray([len(x) for x in selected_by_cell], dtype=np.int64)
    demand = np.bincount(cells.astype(np.int64), minlength=NLIST)
    total_possible = int(query_total_prefix.sum())
    covered = int(current.sum())

    top_alloc = np.argsort(-counts)[:20]
    summary = {
        "budget": budget,
        "min_support": min_support,
        "eligible_candidates": len(eligible),
        "selected_entries": len(selected),
        "selected_nonempty_cells": int(np.count_nonzero(counts)),
        "allocation": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "p95": float(np.quantile(counts, 0.95)),
            "max": int(counts.max()),
            "top_cells": [
                {
                    "cell": int(c),
                    "entries": int(counts[c]),
                    "training_queries": int(demand[c]),
                }
                for c in top_alloc if counts[c] > 0
            ],
        },
        "training_proxy": {
            "total_prefix_io_available": total_possible,
            "covered_prefix_io": covered,
            "covered_fraction": (covered / total_possible) if total_possible else 0.0,
            "mean_predicted_prefix_saved": float(current.mean()),
            "median_predicted_prefix_saved": float(np.median(current)),
            "queries_with_positive_coverage_fraction": float(np.mean(current > 0)),
        },
        "marginal_gain": {
            "first": int(marginal_gains[0]) if marginal_gains else 0,
            "last": int(marginal_gains[-1]) if marginal_gains else 0,
            "median": float(np.median(marginal_gains)) if marginal_gains else 0.0,
        },
    }
    return selected_by_cell, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=10240)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--supports", default="1,2,4")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    q = fbin(args.queries)
    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if len(records) > len(q):
        raise ValueError("more traces than queries")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query numbering mismatch")

    centers, portal_ids, portal_vecs = load_router(args.router)
    cells = route_cells(q[:len(records)], centers, portal_vecs, args.nprobe)

    # Sanity: the first recorded expansion should normally be the static
    # portal selected by the router. Report rather than require 100%, because
    # DiskANN may choose among equal-distance starts differently.
    first_matches = []
    for qi, rec in enumerate(records):
        ids = rec["ids"]
        first_matches.append(bool(ids) and int(ids[0]) == int(portal_ids[cells[qi]]))

    supports = [int(x) for x in args.supports.split(",") if x.strip()]
    manifest = {
        "training_queries": len(records),
        "routing": {
            "nlist": NLIST,
            "nprobe": args.nprobe,
            "cache_key": "static portal winner cell",
            "first_trace_vertex_matches_static_portal_fraction": float(np.mean(first_matches)),
        },
        "objective": "global lazy-greedy marginal coverage of observed expansion-prefix I/O",
        "budget_reference": "CatapultDB paper: 256 LSH buckets * 40 IDs = 10240 IDs",
        "budget_ids": args.budget,
        "variants": {},
    }

    for support in supports:
        selected_by_cell, summary = learn(records, cells, args.budget, support)
        name = f"waypoint-s{support}"
        path = args.out_dir / f"{name}.bin"
        offsets, ids = write_cache(path, selected_by_cell)
        summary["cache_file"] = path.name
        summary["cache_bytes"] = path.stat().st_size
        summary["cache_ids"] = int(len(ids))
        manifest["variants"][name] = summary
        print(name, json.dumps(summary, indent=2), flush=True)

    (args.out_dir / "waypoint-cache-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
