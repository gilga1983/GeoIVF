#!/usr/bin/env python3
"""Learn global ID-only navigation hints from recorded ANN traversals.

For each training query, every vertex contributes at most once, at its first
expansion position. Position zero is the query's teacher start and is excluded.
The default skip-value score is therefore

    score(v) = sum_q first_position_q(v).

The top-B unique database IDs form the NavHints vocabulary. Deployment stores
only uint32 IDs; query-dependent ranking reuses DiskANN's resident PQ codes.

The optional --portals argument exists only for legacy ablations that union a
landmark vocabulary with a released 512-portal set. The core NavHints learner
has no portal dependency.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

START_MAGIC = b"GIDST001"


def load_start_ids(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError(f"{path}: bad start-ID header")
    count, reserved = struct.unpack("<II", raw[8:16])
    expected = 16 + count * 4
    if reserved != 0 or len(raw) != expected:
        raise ValueError(
            f"{path}: bad start-ID size: got {len(raw)}, expected {expected}"
        )
    ids = np.frombuffer(raw, dtype="<u4", count=count, offset=16).copy()
    if count == 0 or len(np.unique(ids)) != count:
        raise ValueError(f"{path}: start IDs must be nonempty and unique")
    return ids


def write_start_ids(path: Path, ids: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(ids), dtype="<u4")
    if len(arr) == 0 or len(np.unique(arr)) != len(arr):
        raise ValueError("start IDs must be nonempty and unique")
    with path.open("wb") as f:
        f.write(START_MAGIC)
        f.write(struct.pack("<II", len(arr), 0))
        arr.tofile(f)
    return arr


def parse_budgets(raw: str) -> list[int]:
    budgets = [int(x) for x in raw.split(",") if x.strip()]
    if not budgets or any(x <= 0 for x in budgets):
        raise ValueError("budgets must be positive integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    return budgets


def load_traces(path: Path, expected_queries: int | None) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"{path}: empty trace")
    if expected_queries is not None and len(records) != expected_queries:
        raise ValueError(
            f"{path}: expected {expected_queries} queries, got {len(records)}"
        )
    expected_ids = list(range(len(records)))
    actual_ids = [int(record["query"]) for record in records]
    if actual_ids != expected_ids:
        raise ValueError(f"{path}: trace query numbering mismatch")
    for qi, record in enumerate(records):
        ids = record.get("ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path}: query {qi} has no traversal IDs")
    return records


def rank_hints(records: list[dict]) -> list[tuple[int, int, int]]:
    score: collections.defaultdict[int, int] = collections.defaultdict(int)
    support: collections.defaultdict[int, int] = collections.defaultdict(int)

    for record in records:
        first_position: dict[int, int] = {}
        for position, raw_id in enumerate(record["ids"]):
            first_position.setdefault(int(raw_id), position)

        for vertex_id, position in first_position.items():
            if position <= 0:
                continue
            score[vertex_id] += position
            support[vertex_id] += 1

    ranked = sorted(
        (
            (int(total), int(support[vertex_id]), int(vertex_id))
            for vertex_id, total in score.items()
            if total > 0
        ),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    if not ranked:
        raise ValueError("trace produced no eligible navigation hints")
    return ranked


def unique_union(first: Iterable[int], second: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in (*list(first), *list(second)):
        value = int(raw)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--budgets",
        default="64,128,256,512,768,1024,1536,2048,2500",
    )
    ap.add_argument(
        "--expected-queries",
        type=int,
        default=5000,
        help="set to 0 to accept any nonempty contiguous trace length",
    )
    ap.add_argument(
        "--portals",
        type=Path,
        help="optional legacy 512-portal union ablation",
    )
    args = ap.parse_args()

    if args.expected_queries < 0:
        raise ValueError("--expected-queries must be nonnegative")
    expected_queries = args.expected_queries or None
    budgets = parse_budgets(args.budgets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_traces(args.trace.resolve(), expected_queries)
    ranked = rank_hints(records)

    portals: list[int] | None = None
    if args.portals is not None:
        portals = list(map(int, load_start_ids(args.portals.resolve())))
        if len(portals) != 512:
            raise ValueError(
                f"legacy portal ablation expects 512 IDs, got {len(portals)}"
            )

    variants: dict[str, dict] = {}
    for budget in budgets:
        if budget > len(ranked):
            raise ValueError(
                f"budget {budget} exceeds {len(ranked)} eligible vertices"
            )

        hints = [vertex_id for _, _, vertex_id in ranked[:budget]]
        hint_path = args.out_dir / f"landmarks-b{budget}.bin"
        hint_ids = write_start_ids(hint_path, hints)
        variants[f"landmarks-b{budget}"] = {
            "landmark_ids": int(len(hint_ids)),
            "state_bytes": hint_path.stat().st_size,
            "file": hint_path.name,
        }

        if portals is not None:
            combined = unique_union(portals, hints)
            combined_path = (
                args.out_dir / f"portals512-plus-landmarks-b{budget}.bin"
            )
            combined_ids = write_start_ids(combined_path, combined)
            variants[f"portals512-plus-landmarks-b{budget}"] = {
                "portal_ids": len(portals),
                "landmark_budget": budget,
                "combined_unique_ids": int(len(combined_ids)),
                "state_bytes": combined_path.stat().st_size,
                "file": combined_path.name,
            }

    result = {
        "training_queries": len(records),
        "score": "sum of first expansion positions across training traversals",
        "eligible_vertices": len(ranked),
        "portal_union_ablation": portals is not None,
        "portal_ids": 0 if portals is None else len(portals),
        "variants": variants,
        "top_landmarks": [
            {"vertex": vertex_id, "skip_score": score, "support": support}
            for score, support, vertex_id in ranked[:20]
        ],
    }
    manifest = args.out_dir / "global-landmarks.manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
