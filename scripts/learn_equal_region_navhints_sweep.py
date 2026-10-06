#!/usr/bin/env python3
"""Build equal-budget regional NavHints variants with one assignment per resolution."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from learn_equal_region_navhints import (
    assign_regions,
    load_fbin,
    load_router,
    write_runtime,
)


def stats(a):
    a = np.asarray(a)
    return {
        "min": int(a.min()) if len(a) else 0,
        "median": float(np.median(a)) if len(a) else 0.0,
        "mean": float(a.mean()) if len(a) else 0.0,
        "p95": float(np.quantile(a, .95)) if len(a) else 0.0,
        "max": int(a.max()) if len(a) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-regions", default="8,16")
    ap.add_argument("--min-entry-pos", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16384)
    args = ap.parse_args()

    budgets = sorted({int(x) for x in args.per_regions.split(",") if x.strip()})
    if not budgets or min(budgets) <= 0:
        raise ValueError("per-region budgets must be positive")
    max_budget = max(budgets)

    nbase, dim, base = load_fbin(args.base)
    nregions, rdim, centers = load_router(args.router)
    if dim != rdim:
        raise ValueError("base/router dimension mismatch")
    labels = assign_regions(base, centers, args.threads, args.batch)

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query IDs must be dense")

    score = [collections.defaultdict(int) for _ in range(nregions)]
    support = [collections.defaultdict(int) for _ in range(nregions)]
    event_count = np.zeros(nregions, dtype=np.int64)

    for rec in records:
        ids = [int(v) for v in rec["ids"]]
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
            for future_vid, p in first_future.items():
                residual = p - pos
                score[region][future_vid] += int(residual)
                support[region][future_vid] += 1

    ranked = []
    for region in range(nregions):
        rr = sorted(
            (
                (int(s), int(support[region][vid]), int(vid))
                for vid, s in score[region].items()
                if s > 0
            ),
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        ranked.append([vid for _, _, vid in rr[:max_budget]])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "regions": nregions,
        "training_queries": len(records),
        "training_events_per_region": stats(event_count),
        "variants": {},
    }
    for budget in budgets:
        selected = [ids[:budget] for ids in ranked]
        out = args.out_dir / f"regional-{nregions}x{budget}.bin"
        offsets, hints = write_runtime(out, labels, selected, args.min_entry_pos)
        counts = np.diff(offsets.astype(np.int64))
        manifest = {
            "regions": nregions,
            "per_region_budget": budget,
            "training_queries": len(records),
            "min_entry_position": args.min_entry_pos,
            "hint_ids": int(len(hints)),
            "hint_ids_per_region": stats(counts),
            "vertex_region_bytes": int(len(labels) * 2),
            "runtime_file_bytes": out.stat().st_size,
            "runtime_file": out.name,
        }
        out.with_suffix(out.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        summary["variants"][f"{nregions}x{budget}"] = manifest

    (args.out_dir / f"regional-{nregions}-sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
