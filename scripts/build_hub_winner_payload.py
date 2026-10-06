#!/usr/bin/env python3
"""Attach independent successful winners to the deployed 16K NavHint hubs.

Inputs:
  * existing 5 x 4 vertex-continuation payload;
  * canonical 16K start-only traces for warm queries;
  * actual L160 top-10 results for the same warm queries.

For every warm query, the first expanded trace ID is its deployed NavHint hub.
The query contributes only its rank-1 returned result. Each hub keeps the first
10 distinct winners it observes chronologically. The output payload appends
three 4-slot variants, giving 12 tail slots; only the first 10 are used.

This is intentionally the simplest online-equivalent learner. No counters,
replacement, distance gate, or multi-result contribution is used.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GIVTX001"
TAIL_VARIANT_TAGS = np.asarray([0xFFFFFFFD, 0xFFFFFFFC, 0xFFFFFFFB], dtype=np.uint32)


def load_payload(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != MAGIC:
        raise ValueError("bad GIVTX001 payload")
    n, variants, slots, min_position = struct.unpack_from("<IIII", raw, 8)
    off = 24
    variant_ids = np.frombuffer(raw, dtype="<u4", count=variants, offset=off).copy()
    off += variants * 4
    count = n * variants * slots
    if len(raw) != off + count * 4:
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


def load_hubs(path: Path, expected: int):
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(records) != expected:
        raise ValueError(f"expected {expected} traces, got {len(records)}")
    hubs = np.empty(expected, dtype=np.uint32)
    for i, r in enumerate(records):
        if int(r["query"]) != i or not r["ids"]:
            raise ValueError("bad trace row")
        hubs[i] = int(r["ids"][0])
    return hubs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-payload", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warm-rows", type=int, default=4000)
    args = ap.parse_args()

    n, variants, slots, min_position, variant_ids, base = load_payload(
        args.base_payload.resolve()
    )
    if variants != 5 or slots != 4:
        raise ValueError(f"expected 5x4 base payload, got {variants}x{slots}")
    results = load_results(args.results.resolve())
    if args.warm_rows <= 0 or args.warm_rows > results.shape[0]:
        raise ValueError("invalid warm row count")
    hubs = load_hubs(args.trace.resolve(), results.shape[0])

    winners_by_hub: dict[int, list[int]] = {}
    seen_by_hub: dict[int, set[int]] = {}
    duplicate_contributions = 0
    full_bucket_contributions = 0

    for i in range(args.warm_rows):
        hub = int(hubs[i])
        winner = int(results[i, 0])
        if hub >= n or winner >= n:
            raise ValueError("ID outside graph")
        bucket = winners_by_hub.setdefault(hub, [])
        seen = seen_by_hub.setdefault(hub, set())
        if winner in seen:
            duplicate_contributions += 1
            continue
        seen.add(winner)
        if len(bucket) < 10:
            bucket.append(winner)
        else:
            full_bucket_contributions += 1

    tail = np.full((n, 12), np.uint32(0xFFFFFFFF), dtype=np.uint32)
    for hub, winners in winners_by_hub.items():
        tail[hub, : len(winners)] = np.asarray(winners, dtype=np.uint32)

    tail_variants = tail.reshape(n, 3, 4)
    combined = np.concatenate([base, tail_variants], axis=1)
    out_variant_ids = np.concatenate([variant_ids, TAIL_VARIANT_TAGS])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", n, variants + 3, slots, min_position))
        out_variant_ids.astype("<u4", copy=False).tofile(f)
        combined.astype("<u4", copy=False).tofile(f)

    counts = np.asarray([len(v) for v in winners_by_hub.values()], dtype=np.int64)
    supports = {}
    for k in (1, 2, 4, 8, 10):
        supports[str(k)] = {
            "hubs_with_at_least_k_winners": int(np.sum(counts >= k)),
            "fraction_active_hubs": float(np.mean(counts >= k)) if len(counts) else 0.0,
        }

    manifest = {
        "policy": "16K deployed hub -> first 10 distinct actual L160 rank-1 winners from different warm searches",
        "warm_rows": args.warm_rows,
        "trace_rows": int(results.shape[0]),
        "vertices": n,
        "base_variants": variants,
        "output_variants": variants + 3,
        "hub_tail_variant_indices": [5, 6, 7],
        "tail_slots_total": 12,
        "tail_slots_used": 10,
        "active_hubs": len(winners_by_hub),
        "winner_count_per_active_hub": {
            "min": int(counts.min()) if len(counts) else 0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "mean": float(counts.mean()) if len(counts) else 0.0,
            "p95": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0,
        },
        "duplicate_contributions_ignored": duplicate_contributions,
        "new_distinct_contributions_after_bucket_full_ignored": full_bucket_contributions,
        "support_by_k": supports,
        "associated_bytes_per_vertex": (variants + 3) * slots * 4,
        "leakage_guard": "only first warm_rows queries contribute; final evaluation suffix does not",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
