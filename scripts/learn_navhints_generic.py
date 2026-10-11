#!/usr/bin/env python3
"""Learn the default NavHints skip-score vocabulary from generic traversal traces."""
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
        raise ValueError("IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(arr), 0))
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=16000)
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace numbering mismatch")

    score = collections.defaultdict(int)
    support = collections.defaultdict(int)
    for rec in records:
        first = {}
        for pos, raw in enumerate(rec["ids"]):
            first.setdefault(int(raw), pos)
        for vid, pos in first.items():
            if pos <= 0:
                continue
            score[vid] += int(pos)
            support[vid] += 1

    ranked = sorted(
        score,
        key=lambda v: (-score[v], -support[v], v),
    )
    used_budget = min(args.budget, len(ranked))
    if used_budget == 0:
        raise ValueError("no eligible navigation landmarks")
    ids = ranked[:used_budget]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ids(args.out, ids)

    result = {
        "training_queries": len(records),
        "score": "sum of first expansion position across training traversals",
        "eligible_vertices": len(ranked),
        "requested_budget": args.budget,
        "landmark_ids": len(ids),
        "budget_capped_by_eligibility": len(ids) < args.budget,
        "state_bytes": args.out.stat().st_size,
        "file": args.out.name,
        "top_landmarks": [
            {"vertex": int(v), "skip_score": int(score[v]), "support": int(support[v])}
            for v in ids[:20]
        ],
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
