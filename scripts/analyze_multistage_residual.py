#!/usr/bin/env python3
"""Screen disjoint stage-specific residual NavHints candidates.

Given an on-policy traversal trace and the deployed stage-0 vocabulary, compute
for stages 1..8:
  * residual-score candidate universe
  * candidates not already in stage 0
  * residual score mass retained after removing stage-0 overlap
  * top-k novel score mass for k in {256,512,1024,2048,4096}

This is a screening diagnostic only. Later-stage candidates are NOT removed
because they were attractive at an earlier stage; doing that would require a
chosen per-stage memory allocation. The output therefore tells us how much
genuinely new signal exists at each checkpoint before choosing that allocation.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

START_MAGIC = b"GIDST001"
TOPKS = (256, 512, 1024, 2048, 4096)


def read_ids(path: Path) -> set[int]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError(f"bad ID file: {path}")
    n, reserved = struct.unpack("<II", raw[8:16])
    if reserved != 0 or len(raw) != 16 + 4 * n:
        raise ValueError(f"bad ID file shape: {path}")
    return set(struct.unpack(f"<{n}I", raw[16:]))


def stage_scores(records, stage: int):
    score = collections.defaultdict(int)
    support = collections.defaultdict(int)
    queries_with_residual = 0
    for rec in records:
        first = {}
        for pos, raw in enumerate(rec["ids"]):
            first.setdefault(int(raw), pos)
        any_residual = False
        for vid, pos in first.items():
            residual = pos - stage
            if residual > 0:
                score[vid] += residual
                support[vid] += 1
                any_residual = True
        queries_with_residual += int(any_residual)
    return score, support, queries_with_residual


def top_mass(items, k: int) -> int:
    return sum(x[0] for x in items[: min(k, len(items))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--start-ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-stage", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    start_ids = read_ids(args.start_ids)

    result = {
        "queries": len(records),
        "stage0_ids": len(start_ids),
        "definition": "S_s(v)=sum_q max(first_expansion_position-s,0)",
        "interpretation": (
            "novel means v is absent from the deployed stage-0 vocabulary; "
            "later stages are screened independently, before choosing per-stage budgets"
        ),
        "topks": list(TOPKS),
        "stages": {},
    }

    for stage in range(1, args.max_stage + 1):
        score, support, qres = stage_scores(records, stage)
        ranked_all = sorted(
            ((total, support[v], v) for v, total in score.items()),
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        ranked_novel = [x for x in ranked_all if x[2] not in start_ids]

        total_mass = sum(x[0] for x in ranked_all)
        novel_mass = sum(x[0] for x in ranked_novel)
        top = {}
        for k in TOPKS:
            mass = top_mass(ranked_novel, k)
            top[str(k)] = {
                "ids": min(k, len(ranked_novel)),
                "score_mass": mass,
                "fraction_of_all_stage_mass": mass / total_mass if total_mass else 0.0,
                "fraction_of_novel_stage_mass": mass / novel_mass if novel_mass else 0.0,
            }

        result["stages"][str(stage)] = {
            "eligible_vertices": len(ranked_all),
            "overlap_with_stage0_vertices": len(ranked_all) - len(ranked_novel),
            "novel_vertices": len(ranked_novel),
            "queries_with_residual": qres,
            "fraction_queries_with_residual": qres / len(records),
            "total_score_mass": total_mass,
            "novel_score_mass": novel_mass,
            "novel_fraction_of_stage_mass": novel_mass / total_mass if total_mass else 0.0,
            "top_novel": top,
            "top20_novel": [
                {"vertex": v, "score": s, "support": sup}
                for s, sup, v in ranked_novel[:20]
            ],
        }

    # Adjacent-stage stability of the best 1K novel candidates.
    for stage in range(1, args.max_stage):
        a = result["stages"][str(stage)]["top20_novel"]
        # Full top-1K sets are recomputed here to keep output compact.
        sa, _, _ = stage_scores(records, stage)
        sb, _, _ = stage_scores(records, stage + 1)
        ra = [
            v for _, _, v in sorted(
                ((t, 0, v) for v, t in sa.items() if v not in start_ids),
                key=lambda x: (-x[0], x[2]),
            )[:1000]
        ]
        rb = [
            v for _, _, v in sorted(
                ((t, 0, v) for v, t in sb.items() if v not in start_ids),
                key=lambda x: (-x[0], x[2]),
            )[:1000]
        ]
        inter = len(set(ra) & set(rb))
        result["stages"][str(stage)]["next_stage_top1k_overlap"] = {
            "intersection": inter,
            "fraction": inter / max(1, min(len(ra), len(rb))),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
