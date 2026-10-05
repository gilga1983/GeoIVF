#!/usr/bin/env python3
"""Learn controlled NavHints landmark-ranking and training-history variants.

All variants consume the same ordinary-DiskANN traversal traces. The emitted
files contain only unique uint32 database IDs in the standard GIDST001 format.

Ranking variants:
  skip        sum of first expansion position across queries (NavHints)
  support     number of training queries containing the vertex
  mean-pos    mean first expansion position when present
  random      deterministic random sample from eligible traversal vertices
  destination historical final-result IDs ranked by result frequency

For history-size experiments, --history-rows restricts every statistic to the
causal prefix [0, history_rows).
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GIDST001"


def write_ids(path: Path, ids):
    arr = np.asarray(ids, dtype="<u4")
    if len(arr) == 0 or len(np.unique(arr)) != len(arr):
        raise ValueError("landmark IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(arr), 0))
        arr.tofile(f)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-rows", type=int, default=5000)
    ap.add_argument("--budgets", default="4096,16000")
    ap.add_argument(
        "--variants",
        default="skip,support,mean-pos,random,destination",
    )
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    args.trace = args.trace.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not all_records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in all_records] != list(range(len(all_records))):
        raise ValueError("trace query numbering mismatch")
    if not 1 <= args.history_rows <= len(all_records):
        raise ValueError("history rows outside trace")
    records = all_records[: args.history_rows]

    skip = collections.defaultdict(int)
    support = collections.defaultdict(int)
    raw_visits = collections.defaultdict(int)
    destinations = collections.Counter()

    for rec in records:
        rid = int(rec.get("result_id", 2**32 - 1))
        if rid != 2**32 - 1:
            destinations[rid] += 1

        first = {}
        for pos, raw in enumerate(rec["ids"]):
            vid = int(raw)
            raw_visits[vid] += 1
            first.setdefault(vid, pos)
        for vid, pos in first.items():
            if pos <= 0:
                continue
            skip[vid] += int(pos)
            support[vid] += 1

    eligible = sorted(skip)
    if not eligible:
        raise ValueError("no eligible traversal landmarks")
    duplicate_visits = sum(raw_visits.values()) - sum(support.values()) - len(records)
    # The subtraction above also removes one pos-0 start per query; it is only
    # a diagnostic, not used in ranking.

    rankings = {}
    rankings["skip"] = sorted(
        eligible,
        key=lambda v: (-skip[v], -support[v], v),
    )
    rankings["support"] = sorted(
        eligible,
        key=lambda v: (-support[v], -skip[v], v),
    )
    rankings["mean-pos"] = sorted(
        eligible,
        key=lambda v: (-(skip[v] / support[v]), -support[v], v),
    )
    rng = np.random.default_rng(args.seed)
    rankings["random"] = list(map(int, rng.permutation(np.asarray(eligible, dtype=np.int64))))
    rankings["destination"] = [
        int(v) for v, _ in sorted(destinations.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in rankings]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if not budgets or any(b <= 0 for b in budgets):
        raise ValueError("budgets must be positive")

    emitted = {}
    for variant in variants:
        rank = rankings[variant]
        for requested in budgets:
            actual = min(requested, len(rank))
            if actual < 512:
                # Our paper router uses at least 512 coarse representatives.
                continue
            ids = rank[:actual]
            out = args.out_dir / f"{variant}-h{args.history_rows}-b{requested}.bin"
            write_ids(out, ids)
            emitted[f"{variant}-b{requested}"] = {
                "variant": variant,
                "history_rows": args.history_rows,
                "requested_budget": requested,
                "actual_ids": actual,
                "budget_saturated": actual == requested,
                "state_bytes": out.stat().st_size,
                "file": out.name,
            }

    result = {
        "history_rows": args.history_rows,
        "trace_rows_available": len(all_records),
        "eligible_traversal_vertices": len(eligible),
        "unique_destinations": len(destinations),
        "raw_visit_events": int(sum(raw_visits.values())),
        "diagnostic_duplicate_visit_events_after_start_adjustment": int(duplicate_visits),
        "definitions": {
            "skip": "sum of first expansion position across training queries",
            "support": "number of training queries containing the vertex after position zero",
            "mean-pos": "mean first expansion position conditional on appearance",
            "random": f"uniform deterministic permutation of eligible vertices; seed={args.seed}",
            "destination": "historical final-result IDs ordered by result frequency",
        },
        "emitted": emitted,
        "top": {
            v: [
                {
                    "vertex": int(x),
                    "skip": int(skip.get(x, 0)),
                    "support": int(support.get(x, 0)),
                    "mean_pos": (
                        float(skip[x] / support[x]) if support.get(x, 0) else None
                    ),
                    "destination_frequency": int(destinations.get(x, 0)),
                }
                for x in rankings[v][:20]
            ]
            for v in variants
        },
    }
    (args.out_dir / f"variants-h{args.history_rows}.manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
