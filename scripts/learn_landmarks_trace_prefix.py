#!/usr/bin/env python3
"""Learn a fixed-size global landmark vocabulary from a prefix of a trace file.

This is intentionally the same skip-position learner as
learn_global_navigation_landmarks.py, except the caller chooses how much causal
history is visible. It is used to compare frozen 5K training against a
progressively warmed cache without changing the scoring rule.
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
    ap.add_argument("--history-rows", type=int, required=True)
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if args.history_rows <= 0 or args.history_rows > len(all_records):
        raise ValueError(
            f"history rows {args.history_rows} outside 1..{len(all_records)}"
        )
    if [int(r["query"]) for r in all_records] != list(range(len(all_records))):
        raise ValueError("trace query numbering mismatch")
    records = all_records[: args.history_rows]

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
    if args.budget > len(ranked):
        raise ValueError(
            f"budget {args.budget} exceeds {len(ranked)} eligible vertices"
        )

    landmarks = [vid for _, _, vid in ranked[: args.budget]]
    path = args.out_dir / f"landmarks-b{args.budget}.bin"
    ids = write_start_ids(path, landmarks)

    # Read portals only as a compatibility/input sanity check. They are not
    # part of the deployed full-cache experiment.
    portals = load_start_ids(args.portals)
    if len(portals) != 512:
        raise ValueError(f"expected 512 portal IDs, got {len(portals)}")

    result = {
        "history_queries": len(records),
        "available_trace_queries": len(all_records),
        "score": "global sum of first expansion positions across visible history",
        "eligible_vertices": len(ranked),
        "landmark_ids": int(len(ids)),
        "state_bytes": int(path.stat().st_size),
        "file": path.name,
        "top_landmarks": [
            {"vertex": vid, "skip_score": s, "support": sup}
            for s, sup, vid in ranked[:20]
        ],
    }
    (args.out_dir / "landmarks.manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
