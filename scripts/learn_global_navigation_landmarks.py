#!/usr/bin/env python3
"""Learn global ID-only navigation landmarks from portal-seeded training traces.

Unlike the regional waypoint table, this deliberately discards the region key.
Each graph vertex is scored across all 5000 training traversals by:
  skip(v) = sum of first expansion position of v.
The top-B unique vertex IDs are emitted for a budget sweep. We also emit each
landmark set unioned with the 512 released portal IDs.

Deployment state is only uint32 IDs and is scored through DiskANN's resident PQ.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

START_MAGIC = b"GIDST001"


def load_start_ids(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError("bad start-ID file")
    count, reserved = struct.unpack("<II", raw[8:16])
    if reserved != 0 or len(raw) != 16 + count * 4:
        raise ValueError("bad start-ID file size")
    return np.frombuffer(raw, dtype="<u4", count=count, offset=16).copy()


def write_start_ids(path: Path, ids):
    ids = np.asarray(ids, dtype="<u4")
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("start IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(START_MAGIC)
        f.write(struct.pack("<II", len(ids), 0))
        ids.tofile(f)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--portals", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budgets", default="64,128,256,512,768,1024,1536,2048,2500")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if len(records) != 5000:
        raise ValueError(f"expected 5000 training traces, got {len(records)}")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query numbering mismatch")

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
        (
            (int(s), int(support[vid]), int(vid))
            for vid, s in score.items()
            if s > 0
        ),
        key=lambda x: (-x[0], -x[1], x[2]),
    )
    portals = list(map(int, load_start_ids(args.portals)))
    if len(portals) != 512:
        raise ValueError(f"expected 512 released portals, got {len(portals)}")

    variants = {}
    for budget in [int(x) for x in args.budgets.split(",") if x.strip()]:
        if budget > len(ranked):
            raise ValueError(f"budget {budget} exceeds {len(ranked)} candidates")
        landmarks = [vid for _, _, vid in ranked[:budget]]

        lp = args.out_dir / f"landmarks-b{budget}.bin"
        lids = write_start_ids(lp, landmarks)
        variants[f"landmarks-b{budget}"] = {
            "landmark_ids": int(len(lids)),
            "state_bytes": lp.stat().st_size,
            "file": lp.name,
        }

        combined = []
        seen = set()
        for v in [*portals, *landmarks]:
            if v not in seen:
                seen.add(v)
                combined.append(v)
        cp = args.out_dir / f"portals512-plus-landmarks-b{budget}.bin"
        cids = write_start_ids(cp, combined)
        variants[f"portals512-plus-landmarks-b{budget}"] = {
            "portal_ids": 512,
            "landmark_budget": budget,
            "combined_unique_ids": int(len(cids)),
            "state_bytes": cp.stat().st_size,
            "file": cp.name,
        }

    result = {
        "training_queries": len(records),
        "score": "global sum of first expansion positions across training traversals",
        "eligible_vertices": len(ranked),
        "portal_ids": 512,
        "variants": variants,
        "top_landmarks": [
            {"vertex": vid, "skip_score": s, "support": sup}
            for s, sup, vid in ranked[:20]
        ],
    }
    (args.out_dir / "global-landmarks.manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
