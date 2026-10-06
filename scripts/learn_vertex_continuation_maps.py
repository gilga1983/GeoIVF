#!/usr/bin/env python3
"""Learn vertex-granular continuation maps with regional backoff.

For each on-policy training trace, positions before --min-position are treated as
bootstrap and ignored. Every later visited source vertex accumulates residual
skip score for unique vertices that appear after it:

    score_v(u) += first_future_position(u) - position(v)

Support(v) is the number of training traces in which v appears at or after the
minimum position. To avoid materializing huge pair tables for one-off vertices,
we first count support and only learn local maps for vertices with support >=
the smallest requested threshold.

Deployment keeps exactly four continuation IDs per vertex. For a threshold t:
  * support(v) >= t: use v's learned top continuation IDs, filling any empty
    slots from v's regional map;
  * otherwise: use the regional map directly.

The output packs the pure regional baseline plus all support thresholds into one
file, so a single enriched DiskANN graph can evaluate every learning resolution.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

REGIONAL_MAGIC = b"GIRGN001"
OUTPUT_MAGIC = b"GIVTX001"
SENTINEL = 0xFFFFFFFF


def parse_regional(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != REGIONAL_MAGIC:
        raise ValueError("bad regional file")
    nvertices, nregions, total, min_entry_pos = struct.unpack_from("<IIII", raw, 8)
    off = 24
    labels = np.frombuffer(raw, dtype="<u2", count=nvertices, offset=off).copy()
    off += nvertices * 2
    offsets = np.frombuffer(raw, dtype="<u4", count=nregions + 1, offset=off).copy()
    off += (nregions + 1) * 4
    hint_ids = np.frombuffer(raw, dtype="<u4", count=total, offset=off).copy()
    off += total * 4
    if off != len(raw):
        raise ValueError("regional file size mismatch")
    if offsets[0] != 0 or offsets[-1] != total or np.any(offsets[:-1] > offsets[1:]):
        raise ValueError("invalid regional offsets")
    if np.any(labels >= nregions):
        raise ValueError("regional label outside range")
    return int(nvertices), int(nregions), int(min_entry_pos), labels, offsets, hint_ids


def regional_map(vid, labels, offsets, hint_ids, slots):
    region = int(labels[vid])
    lo, hi = int(offsets[region]), int(offsets[region + 1])
    out = []
    seen = {int(vid)}
    for raw in hint_ids[lo:hi]:
        x = int(raw)
        if x == SENTINEL or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) == slots:
            break
    out.extend([SENTINEL] * (slots - len(out)))
    return out


def qstats(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": int(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, .95)),
        "max": int(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--regional", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", default="2,4,8,16")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--min-position", type=int, default=8)
    args = ap.parse_args()

    thresholds = sorted({int(x) for x in args.thresholds.split(",") if x.strip()})
    if not thresholds or min(thresholds) <= 0:
        raise ValueError("thresholds must be positive")
    if args.slots <= 0:
        raise ValueError("slots must be positive")

    nvertices, nregions, regional_min_pos, labels, offsets, regional_ids = parse_regional(
        args.regional
    )
    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query IDs must be dense")

    support = np.zeros(nvertices, dtype=np.uint32)
    eligible_min = thresholds[0]

    # Pass 1: trace-level support after bootstrap.
    for rec in records:
        ids = [int(v) for v in rec["ids"]]
        if any(v < 0 or v >= nvertices for v in ids):
            raise ValueError("trace vertex outside graph")
        seen = set()
        for v in ids[args.min_position:]:
            if v not in seen:
                support[v] += 1
                seen.add(v)

    eligible = set(np.flatnonzero(support >= eligible_min).tolist())
    print(
        json.dumps({
            "pass1_eligible_vertices": len(eligible),
            "min_support": eligible_min,
            "support_nonzero": int(np.count_nonzero(support)),
        }),
        flush=True,
    )

    # Pass 2: downstream score only for vertices that could specialize in any arm.
    score = {v: collections.defaultdict(int) for v in eligible}
    pair_support = {v: collections.defaultdict(int) for v in eligible}
    pair_updates = 0

    for qi, rec in enumerate(records):
        ids = [int(v) for v in rec["ids"]]
        # Traversal IDs are normally unique, but keep first occurrence semantics.
        first_pos = {}
        for p, v in enumerate(ids):
            first_pos.setdefault(v, p)
        ordered = sorted(first_pos.items(), key=lambda x: x[1])

        for source, p in ordered:
            if p < args.min_position or source not in eligible:
                continue
            seen_future = set()
            for target, tp in ordered:
                if tp <= p or target == source or target in seen_future:
                    continue
                seen_future.add(target)
                residual = tp - p
                score[source][target] += int(residual)
                pair_support[source][target] += 1
                pair_updates += 1

        if qi == 0 or (qi + 1) % 1000 == 0 or qi + 1 == len(records):
            print(f"scored_traces={qi+1}/{len(records)} pair_updates={pair_updates}", flush=True)

    local_top = {}
    for v in eligible:
        ranked = sorted(
            (
                (int(s), int(pair_support[v][u]), int(u))
                for u, s in score[v].items()
                if s > 0 and u != v
            ),
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        local_top[v] = [u for _, _, u in ranked[: args.slots]]

    variants = [0] + thresholds
    nvariants = len(variants)
    payload = np.full((nvertices, nvariants, args.slots), SENTINEL, dtype="<u4")
    regional_cache = {}

    def fallback(v):
        r = int(labels[v])
        cached = regional_cache.get((r, v))
        if cached is None:
            cached = regional_map(v, labels, offsets, regional_ids, args.slots)
            regional_cache[(r, v)] = cached
        return cached

    # Regional baseline.
    for v in range(nvertices):
        payload[v, 0, :] = fallback(v)

    coverage = {}
    for vi, threshold in enumerate(thresholds, start=1):
        specialized = 0
        fully_local = 0
        for v in range(nvertices):
            base = fallback(v)
            if int(support[v]) < threshold:
                payload[v, vi, :] = base
                continue
            specialized += 1
            out = []
            seen = {v}
            for u in local_top.get(v, ()):
                if u == SENTINEL or u in seen:
                    continue
                seen.add(u)
                out.append(u)
                if len(out) == args.slots:
                    break
            if len(out) == args.slots:
                fully_local += 1
            if len(out) < args.slots:
                for u in base:
                    if u == SENTINEL or u in seen:
                        continue
                    seen.add(u)
                    out.append(int(u))
                    if len(out) == args.slots:
                        break
            out = out[: args.slots]
            out.extend([SENTINEL] * (args.slots - len(out)))
            payload[v, vi, :] = out

        coverage[str(threshold)] = {
            "specialized_vertices": specialized,
            "specialized_fraction": specialized / nvertices,
            "fully_local_vertices": fully_local,
            "fully_local_fraction": fully_local / nvertices,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(OUTPUT_MAGIC)
        f.write(struct.pack("<IIII", nvertices, nvariants, args.slots, args.min_position))
        np.asarray(variants, dtype="<u4").tofile(f)
        payload.tofile(f)

    nonzero = support[support > 0]
    manifest = {
        "format": OUTPUT_MAGIC.decode(),
        "training_queries": len(records),
        "vertices": nvertices,
        "regions": nregions,
        "regional_source_min_entry_position": regional_min_pos,
        "min_training_position": args.min_position,
        "slots_per_variant": args.slots,
        "variant_values": variants,
        "variant_semantics": {
            "0": "pure 256-region fallback",
            **{str(t): f"per-vertex map when support >= {t}, region fallback otherwise" for t in thresholds},
        },
        "support_nonzero_vertices": int(np.count_nonzero(support)),
        "support_nonzero_fraction": float(np.count_nonzero(support) / nvertices),
        "support_distribution_nonzero": qstats(nonzero),
        "eligible_vertices_for_pair_learning": len(eligible),
        "pair_updates": int(pair_updates),
        "coverage": coverage,
        "score": "sum of residual downstream expansion distance after source vertex",
        "backoff": "fill missing local slots with source vertex's 256-region map",
        "bytes": args.out.stat().st_size,
        "output": args.out.name,
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
