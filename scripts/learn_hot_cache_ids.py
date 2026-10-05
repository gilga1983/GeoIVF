#!/usr/bin/env python3
"""Learn workload-hot full-node cache IDs from DiskANN traversal traces.

Unlike NavHints, these IDs are not used as graph starts. The deployment patch
preloads each selected node's full graph record into DiskANN's existing static
cache. Ranking uses training-query support (number of traversals containing the
vertex), with total first-position skip value as deterministic secondary score.
Position zero is INCLUDED because a hot cache should be allowed to cache the
ordinary medoid if training evidence says it is hot.
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
        raise ValueError("cache IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(arr), 0))
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--counts", default="16,24,32,48,64,128,256,512,900")
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace numbering mismatch")

    support = collections.defaultdict(int)
    skip = collections.defaultdict(int)
    for rec in records:
        first = {}
        for pos, raw in enumerate(rec["ids"]):
            first.setdefault(int(raw), pos)
        for vid, pos in first.items():
            support[vid] += 1
            skip[vid] += int(pos)

    ranked = sorted(
        support,
        key=lambda v: (-support[v], -skip[v], v),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for count in [int(x) for x in args.counts.split(",") if x.strip()]:
        if not 0 < count <= len(ranked):
            raise ValueError(f"invalid count {count}; eligible {len(ranked)}")
        out = args.out_dir / f"hot-cache-n{count}.bin"
        ids = ranked[:count]
        write_ids(out, ids)
        variants[str(count)] = {
            "nodes": count,
            "file": out.name,
            "id_file_bytes": out.stat().st_size,
            "top_support_min": int(support[ids[-1]]),
        }

    result = {
        "training_queries": len(records),
        "ranking": (
            "number of training traversals containing vertex; "
            "secondary sum of first expansion position"
        ),
        "position_zero_included": True,
        "eligible_vertices": len(ranked),
        "variants": variants,
        "top_nodes": [
            {
                "vertex": int(v),
                "support": int(support[v]),
                "skip_sum": int(skip[v]),
            }
            for v in ranked[:20]
        ],
    }
    (args.out_dir / "hot-cache.manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
