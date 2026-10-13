#!/usr/bin/env python3
"""Controlled alternatives to NavHints' learned junction score.

All methods use the SAME frozen teacher-trace candidate universe:
vertices first expanded after position zero in at least one training query.
Only the ranking changes. The position-sum arm reproduces the existing
learner byte for byte; no default-paper algorithm is modified.

Output: OUT/{position-sum,frequency,random}/landmarks-bB.bin
        and a provenance/overlap manifest.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from learn_global_navigation_landmarks import write_start_ids, load_start_ids


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_from_trace(trace: Path, *, random_seed: int):
    scores = collections.defaultdict(int)
    visits = collections.defaultdict(int)
    requests = 0
    with trace.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["query"]) != requests:
                raise ValueError("teacher trace query indices are not contiguous")
            requests += 1
            first = {}
            for position, raw in enumerate(row["ids"]):
                first.setdefault(int(raw), position)
            for vid, position in first.items():
                if position > 0:
                    scores[vid] += position
                    visits[vid] += 1
    if requests == 0:
        raise ValueError("empty teacher trace")

    eligible = sorted(scores)
    position_sum = sorted(eligible, key=lambda v: (-scores[v], -visits[v], v))
    frequency = sorted(eligible, key=lambda v: (-visits[v], v))
    random_ids = list(eligible)
    random.Random(random_seed).shuffle(random_ids)
    return requests, eligible, {
        "position-sum": position_sum,
        "frequency": frequency,
        "random": random_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=14464)
    ap.add_argument("--random-seed", type=int, default=20261013)
    ap.add_argument("--published-landmarks", type=Path,
                    help="Existing frozen position-sum landmarks: must match byte for byte.")
    a = ap.parse_args()
    a.trace = a.trace.resolve()
    if a.budget <= 0:
        ap.error("budget must be positive")
    requests, eligible, rankings = rank_from_trace(a.trace, random_seed=a.random_seed)
    if len(eligible) < a.budget:
        raise ValueError(f"only {len(eligible)} eligible vertices, budget={a.budget}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    chosen = {}
    output = {}
    for mode, ordered in rankings.items():
        ids = ordered[:a.budget]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{mode}: repeated vertex IDs")
        d = a.out_dir / mode
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"landmarks-b{a.budget}.bin"
        write_start_ids(out, ids)
        if list(map(int, load_start_ids(out))) != ids:
            raise ValueError(f"{mode}: output round-trip mismatch")
        chosen[mode] = set(ids)
        output[mode] = {
            "path": str(out.resolve()),
            "sha256": digest(out),
            "budget": len(ids),
            "first_16_ids": ids[:16],
        }

    if a.published_landmarks:
        old = a.published_landmarks.resolve()
        if old.read_bytes() != (a.out_dir / "position-sum" / f"landmarks-b{a.budget}.bin").read_bytes():
            raise RuntimeError("position-sum control does not reproduce frozen paper landmarks")
        output["position-sum"]["published_sha256"] = digest(old)

    overlap = {
        mode: {
            "shared_with_position_sum": len(chosen[mode] & chosen["position-sum"]),
            "jaccard_with_position_sum": (
                len(chosen[mode] & chosen["position-sum"]) /
                len(chosen[mode] | chosen["position-sum"])
            ),
        }
        for mode in rankings
    }
    result = {
        "experiment": "junction-score-ablation",
        "training_trace": str(a.trace),
        "training_trace_sha256": digest(a.trace),
        "teacher_requests": requests,
        "eligible_vertex_count": len(eligible),
        "excludes_initial_expansion_position": True,
        "fixed_id_budget": a.budget,
        "random_seed": a.random_seed,
        "definitions": {
            "position-sum": "sum of first expansion positions over training queries, descending; tie break support then vertex ID",
            "frequency": "number of training queries expanding vertex after initial position, descending; tie break vertex ID",
            "random": "uniform random permutation of the same eligible vertices with fixed seed",
        },
        "arms": output,
        "overlap": overlap,
    }
    (a.out_dir / "score-controls.manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
