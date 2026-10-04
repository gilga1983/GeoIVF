#!/usr/bin/env python3
"""Learn simple navigation-heavy-hitter starting points from DiskANN traces.

For each query routed to a static portal cell, count which graph vertices were
expanded during the search. Two intentionally simple scores are produced:

  freq: number of training queries in the cell whose traversal contains v.
  skip: sum of v's first expansion position across those queries.

"skip" rewards vertices that are both recurrent and reached later, since they
could bypass a longer search prefix when reused as a start.

We emit both fixed per-cell top-X policies and global-budget policies.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from learn_waypoint_cache import (
    NLIST,
    fbin,
    load_router,
    route_cells,
    write_cache,
)


def collect(records, cells):
    freq = defaultdict(int)
    skip = defaultdict(int)
    for qi, rec in enumerate(records):
        cell = int(cells[qi])
        first = {}
        for pos, raw in enumerate(rec["ids"]):
            first.setdefault(int(raw), pos)
        for vid, pos in first.items():
            key = (cell, vid)
            freq[key] += 1
            skip[key] += int(pos)
    return freq, skip


def fixed_per_cell(scores, topx):
    by_cell = [[] for _ in range(NLIST)]
    for (cell, vid), score in scores.items():
        by_cell[cell].append((int(score), int(vid)))
    selected = [[] for _ in range(NLIST)]
    for cell, vals in enumerate(by_cell):
        vals.sort(key=lambda x: (-x[0], x[1]))
        selected[cell] = [vid for score, vid in vals[:topx] if score > 0]
    return selected


def global_budget(scores, budget):
    ranked = sorted(
        ((int(score), int(cell), int(vid)) for (cell, vid), score in scores.items() if score > 0),
        key=lambda x: (-x[0], x[1], x[2]),
    )[:budget]
    selected = [[] for _ in range(NLIST)]
    for score, cell, vid in ranked:
        selected[cell].append(vid)
    return selected


def summarize(selected, demand):
    counts = np.asarray([len(x) for x in selected], dtype=np.int64)
    top = np.argsort(-counts)[:20]
    return {
        "selected_entries": int(counts.sum()),
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
                for c in top if counts[c] > 0
            ],
        },
    }


def emit(out_dir, name, selected, demand, metadata):
    path = out_dir / f"{name}.bin"
    _, ids = write_cache(path, selected)
    s = summarize(selected, demand)
    s.update(metadata)
    s["cache_file"] = path.name
    s["cache_ids"] = int(len(ids))
    s["cache_bytes"] = path.stat().st_size
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--topx", default="5,10,20,40")
    ap.add_argument("--budgets", default="2000,2500,2800")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    q = fbin(args.queries)
    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query numbering mismatch")

    centers, portal_ids, portal_vecs = load_router(args.router)
    cells = route_cells(q[:len(records)], centers, portal_vecs, args.nprobe)
    demand = np.bincount(cells.astype(np.int64), minlength=NLIST)
    freq, skip = collect(records, cells)

    variants = {}
    for topx in [int(x) for x in args.topx.split(",") if x.strip()]:
        for score_name, scores in (("freq", freq), ("skip", skip)):
            name = f"heavy-local-{score_name}-x{topx}"
            selected = fixed_per_cell(scores, topx)
            variants[name] = emit(
                args.out_dir,
                name,
                selected,
                demand,
                {
                    "policy": "per-cell navigation heavy hitters",
                    "score": score_name,
                    "topx_per_cell": topx,
                },
            )

    for budget in [int(x) for x in args.budgets.split(",") if x.strip()]:
        for score_name, scores in (("freq", freq), ("skip", skip)):
            name = f"heavy-global-{score_name}-b{budget}"
            selected = global_budget(scores, budget)
            variants[name] = emit(
                args.out_dir,
                name,
                selected,
                demand,
                {
                    "policy": "global navigation heavy hitters",
                    "score": score_name,
                    "budget_ids": budget,
                },
            )

    result = {
        "training_queries": len(records),
        "routing": {
            "nlist": NLIST,
            "nprobe": args.nprobe,
            "cache_key": "static portal winner cell",
        },
        "scores": {
            "freq": "number of training-query traversals containing the vertex",
            "skip": "sum of first expansion positions; recurrent later vertices score higher",
        },
        "variants": variants,
    }
    (args.out_dir / "navigation-heavy-hitters-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
