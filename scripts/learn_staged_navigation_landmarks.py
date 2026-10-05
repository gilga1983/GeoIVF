#!/usr/bin/env python3
"""Learn equal-budget stage-specific NavHints vocabularies from traversal traces.

For stage s, vertex v receives residual skip value:
    S_s(v) = sum_q max(p_q(v) - s, 0)
where p_q(v) is v's first expansion position in query q.

Stage 0 is the ordinary NavHints score. Stage 8 values only residual progress
remaining after eight expansions. The stage vocabularies are learned
independently; overlap is measured rather than artificially removed.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

START_MAGIC = b"GIDST001"


def write_start_ids(path: Path, ids):
    ids = np.asarray(ids, dtype="<u4")
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(START_MAGIC)
        f.write(struct.pack("<II", len(ids), 0))
        ids.tofile(f)


def rank_stage(records, stage: int):
    score = collections.defaultdict(int)
    support = collections.defaultdict(int)
    for rec in records:
        first = {}
        for pos, raw in enumerate(rec["ids"]):
            first.setdefault(int(raw), pos)
        for vid, pos in first.items():
            residual = int(pos) - stage
            if residual <= 0:
                continue
            score[vid] += residual
            support[vid] += 1

    ranked = sorted(
        (
            (int(total), int(support[vid]), int(vid))
            for vid, total in score.items()
            if total > 0
        ),
        key=lambda x: (-x[0], -x[1], x[2]),
    )
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--stages", default="0,8")
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if len(records) != 5000:
        raise ValueError(f"expected 5000 training traces, got {len(records)}")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query numbering mismatch")
    if args.budget <= 0:
        raise ValueError("budget must be positive")

    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    if not stages or any(s < 0 for s in stages):
        raise ValueError("stages must be nonnegative")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    chosen = {}

    for stage in stages:
        ranked = rank_stage(records, stage)
        if args.budget > len(ranked):
            raise ValueError(
                f"budget {args.budget} exceeds {len(ranked)} eligible vertices at stage {stage}"
            )
        ids = [vid for _, _, vid in ranked[: args.budget]]
        path = args.out_dir / f"stage{stage}-b{args.budget}.bin"
        write_start_ids(path, ids)
        chosen[stage] = set(ids)
        variants[str(stage)] = {
            "stage": stage,
            "score": f"sum max(first_expansion_position - {stage}, 0)",
            "eligible_vertices": len(ranked),
            "landmark_ids": len(ids),
            "state_bytes": path.stat().st_size,
            "file": path.name,
            "top_landmarks": [
                {"vertex": vid, "residual_skip_score": total, "support": support}
                for total, support, vid in ranked[:20]
            ],
        }

    overlap = {}
    for i, s1 in enumerate(stages):
        for s2 in stages[i + 1 :]:
            inter = len(chosen[s1] & chosen[s2])
            union = len(chosen[s1] | chosen[s2])
            overlap[f"{s1}-{s2}"] = {
                "intersection": inter,
                "fraction_of_each_budget": inter / args.budget,
                "jaccard": inter / union if union else 0.0,
            }

    result = {
        "training_queries": len(records),
        "budget_per_stage": args.budget,
        "stages": stages,
        "definition": "S_s(v)=sum_q max(p_q(v)-s,0), p_q(v)=first expansion position",
        "variants": variants,
        "overlap": overlap,
    }
    (args.out_dir / "staged-landmarks.manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
