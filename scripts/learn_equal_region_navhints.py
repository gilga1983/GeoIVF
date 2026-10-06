#!/usr/bin/env python3
"""Learn equal-budget per-region NavHints and a runtime vertex->region table.

The file is intentionally deployment-oriented.  It stores:
  * one uint16 region id for every database vertex;
  * one fixed-size (or smaller, if evidence is sparse) ID list per region.

Training is on traversal traces from the deployed start policy.  For each query,
the first entry into a previously unseen region at position >= --min-entry-pos
creates a regional navigation event.  Every later unique vertex receives
residual skip score (later_position - entry_position).  Each region keeps its
own top --per-region candidates, so the memory allocation is deliberately
simple and equal across regions.

Runtime format GIRGN001:
  magic[8]
  uint32 nvertices, uint32 nregions, uint32 total_hint_ids, uint32 min_entry_pos
  uint16 vertex_region[nvertices]
  uint32 offsets[nregions+1]
  uint32 hint_ids[total_hint_ids]
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import faiss
import numpy as np

ROUTER_MAGIC = b"GIPIP001"
MAGIC = b"GIRGN001"


def load_fbin(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated fbin: {path}")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"bad fbin size: {path}")
    return rows, dim, np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_router(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != ROUTER_MAGIC:
        raise ValueError("bad router header")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nf = nlist * dim
    off = 16
    centers = np.frombuffer(raw, dtype="<f4", count=nf, offset=off).reshape(nlist, dim).copy()
    off += nf * 4
    off += nlist * 4  # portal ids
    off += nf * 4     # portal vectors
    if off != len(raw):
        raise ValueError("router trailing bytes")
    if nlist > np.iinfo(np.uint16).max:
        raise ValueError("region count exceeds uint16 runtime encoding")
    return int(nlist), int(dim), centers


def assign_regions(base, centers, threads: int, batch: int):
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(centers.shape[1])
    index.add(np.asarray(centers, dtype=np.float32, order="C"))
    labels = np.empty(len(base), dtype="<u2")
    for lo in range(0, len(base), batch):
        hi = min(len(base), lo + batch)
        x = np.asarray(base[lo:hi], dtype=np.float32, order="C")
        _, ids = index.search(x, 1)
        if np.any(ids < 0):
            raise RuntimeError("FAISS region assignment returned invalid cell")
        labels[lo:hi] = ids[:, 0].astype("<u2", copy=False)
        if lo == 0 or hi == len(base) or (lo // batch) % 8 == 0:
            print(f"assigned_regions={hi}/{len(base)}", flush=True)
    return labels


def write_runtime(path: Path, labels, selected, min_entry_pos: int):
    nvertices = len(labels)
    nregions = len(selected)
    offsets = np.zeros(nregions + 1, dtype="<u4")
    flat = []
    for r, ids in enumerate(selected):
        seen = set()
        for raw in ids:
            v = int(raw)
            if v not in seen:
                seen.add(v)
                flat.append(v)
        offsets[r + 1] = len(flat)
    hints = np.asarray(flat, dtype="<u4")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", nvertices, nregions, len(hints), min_entry_pos))
        np.asarray(labels, dtype="<u2").tofile(f)
        offsets.tofile(f)
        hints.tofile(f)
    return offsets, hints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-region", type=int, default=16)
    ap.add_argument("--min-entry-pos", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16384)
    args = ap.parse_args()

    if args.per_region <= 0 or args.min_entry_pos < 0:
        raise ValueError("invalid per-region/min-entry-pos")
    nbase, dim, base = load_fbin(args.base)
    nregions, rdim, centers = load_router(args.router)
    if dim != rdim:
        raise ValueError("base/router dimension mismatch")

    labels = assign_regions(base, centers, args.threads, args.batch)

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query IDs must be dense from zero")

    score = [collections.defaultdict(int) for _ in range(nregions)]
    support = [collections.defaultdict(int) for _ in range(nregions)]
    event_count = np.zeros(nregions, dtype=np.int64)
    total_events = 0

    for rec in records:
        ids = [int(v) for v in rec["ids"]]
        if any(v < 0 or v >= nbase for v in ids):
            raise ValueError("trace vertex outside database")
        seen_regions = set()
        for pos, vid in enumerate(ids):
            region = int(labels[vid])
            if region in seen_regions:
                continue
            seen_regions.add(region)
            if pos < args.min_entry_pos or pos + 1 >= len(ids):
                continue

            first_future = {}
            for p in range(pos + 1, len(ids)):
                first_future.setdefault(ids[p], p)
            if not first_future:
                continue

            event_count[region] += 1
            total_events += 1
            for future_vid, p in first_future.items():
                residual = p - pos
                score[region][future_vid] += int(residual)
                support[region][future_vid] += 1

    selected = []
    for region in range(nregions):
        ranked = sorted(
            (
                (int(s), int(support[region][vid]), int(vid))
                for vid, s in score[region].items()
                if s > 0
            ),
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        selected.append([vid for _, _, vid in ranked[: args.per_region]])

    offsets, hints = write_runtime(args.out, labels, selected, args.min_entry_pos)
    counts = np.diff(offsets.astype(np.int64))
    manifest = {
        "format": MAGIC.decode(),
        "training_queries": len(records),
        "database_vertices": nbase,
        "dimension": dim,
        "regions": nregions,
        "per_region_budget": args.per_region,
        "min_entry_position": args.min_entry_pos,
        "event_definition": "first entry into a previously unseen region at or after min_entry_position",
        "score": "sum of residual first-future expansion distance from the regional entry",
        "training_events": int(total_events),
        "training_events_per_region": {
            "min": int(event_count.min()),
            "median": float(np.median(event_count)),
            "mean": float(event_count.mean()),
            "p95": float(np.quantile(event_count, .95)),
            "max": int(event_count.max()),
        },
        "hint_ids": int(len(hints)),
        "hint_ids_per_region": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "p95": float(np.quantile(counts, .95)),
            "max": int(counts.max()),
        },
        "vertex_region_bytes": int(len(labels) * 2),
        "runtime_file_bytes": args.out.stat().st_size,
        "runtime_file": args.out.name,
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
