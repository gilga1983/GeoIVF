#!/usr/bin/env python3
"""Append a persistent online co-result overlay to the packed vertex-hint payload.

The first five variants are copied exactly from the existing GIVTX001 payload.
A sixth 4-ID variant is learned only from the warm prefix of actual completed
DiskANN searches. For each warm result set:
  anchor = returned rank-1 ID
  hints  = returned ranks 2..5 (distinct)
and the anchor's persistent overlay is overwritten by the latest observation.

No evaluation-query results are used. This isolates persistence itself before
testing batching, spreading, support counters, or alternative update rules.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GIVTX001"
PERSISTENT_VARIANT_TAG = np.uint32(0xFFFFFFFE)


def load_payload(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != MAGIC:
        raise ValueError("bad GIVTX001 payload")
    n, variants, slots, min_position = struct.unpack_from("<IIII", raw, 8)
    off = 24
    variant_ids = np.frombuffer(raw, dtype="<u4", count=variants, offset=off).copy()
    off += variants * 4
    count = n * variants * slots
    expected = off + count * 4
    if len(raw) != expected:
        raise ValueError("payload size mismatch")
    data = np.frombuffer(raw, dtype="<u4", count=count, offset=off).reshape(
        n, variants, slots
    ).copy()
    return int(n), int(variants), int(slots), int(min_position), variant_ids, data


def load_results(path: Path):
    with path.open("rb") as f:
        rows, k = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows * k)
    if a.size != rows * k:
        raise ValueError("truncated result dump")
    return a.reshape(rows, k)


def distinct(row):
    out = []
    seen = set()
    for raw in row:
        x = int(raw)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-payload", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warm-rows", type=int, default=4000)
    args = ap.parse_args()

    n, variants, slots, min_position, variant_ids, base = load_payload(
        args.base_payload.resolve()
    )
    if slots != 4:
        raise ValueError("expected four slots per packed variant")
    results = load_results(args.results.resolve())
    if args.warm_rows <= 0 or args.warm_rows > results.shape[0]:
        raise ValueError("invalid warm row count")

    persistent = np.full((n, slots), np.uint32(0xFFFFFFFF), dtype=np.uint32)
    writes = 0
    overwrites = 0
    unique_anchors = set()
    missing_siblings = 0

    for row in results[: args.warm_rows]:
        ids = distinct(row)
        if len(ids) < 2:
            continue
        anchor = ids[0]
        if anchor < 0 or anchor >= n:
            raise ValueError("anchor outside graph")
        siblings = ids[1 : 1 + slots]
        if anchor in unique_anchors:
            overwrites += 1
        unique_anchors.add(anchor)
        persistent[anchor].fill(np.uint32(0xFFFFFFFF))
        if siblings:
            persistent[anchor, : len(siblings)] = np.asarray(siblings, dtype=np.uint32)
        if len(siblings) < slots:
            missing_siblings += 1
        writes += 1

    combined = np.concatenate([base, persistent[:, None, :]], axis=1)
    out_variants = np.concatenate([variant_ids, np.asarray([PERSISTENT_VARIANT_TAG], dtype=np.uint32)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", n, variants + 1, slots, min_position))
        out_variants.astype("<u4", copy=False).tofile(f)
        combined.astype("<u4", copy=False).tofile(f)

    nonempty = np.any(persistent != np.uint32(0xFFFFFFFF), axis=1)
    manifest = {
        "base_payload": str(args.base_payload.resolve()),
        "results": str(args.results.resolve()),
        "output": str(args.output.resolve()),
        "vertices": n,
        "base_variants": variants,
        "output_variants": variants + 1,
        "persistent_variant_index": variants,
        "slots": slots,
        "warm_rows": args.warm_rows,
        "writes": writes,
        "unique_persistent_anchors": int(nonempty.sum()),
        "fraction_vertices_persistent": float(nonempty.mean()),
        "overwrites_latest_wins": overwrites,
        "missing_sibling_rows": missing_siblings,
        "persistent_bytes_per_vertex": slots * 4,
        "combined_associated_bytes_per_vertex": (variants + 1) * slots * 4,
        "leakage_guard": "only result rows before evaluation offset are used",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
