#!/usr/bin/env python3
"""Derive controlled NavHints junction-score variants from the SAME trace.

Variants:
  sum  : sum_q(first_expansion_position_q(v)), selected paper heuristic
  count: number of training queries in which v was expanded
  mean : sum_q(first_expansion_position_q(v)) / count(v)

All exclude the initial starting position and deduplicate a vertex within a
query. The retained B=14464 IDs are grouped with the exact same 512-region
offline builder and seed in the runner; no algorithm code is changed.
Fail closed unless the sum variant matches the frozen original landmark file.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

MAGIC = b"GIDST001"
MODES = ("sum", "count", "mean")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Junction ID list empty or contains duplicates")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(ids), 0))
        for vid in ids:
            if not 0 <= vid < (1 << 32):
                raise ValueError("ID outside u32 range")
            f.write(struct.pack("<I", vid))


def select(stats, mode, budget):
    support, position = stats
    if mode == "sum":
        return sorted(position, key=lambda vid: (-position[vid], -support[vid], vid))[
            :budget
        ]
    if mode == "count":
        return sorted(support, key=lambda vid: (-support[vid], vid))[:budget]
    if mode == "mean":
        return sorted(
            position,
            key=lambda vid: (-position[vid] / support[vid], -support[vid], vid),
        )[:budget]
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--reference-sum", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget", type=int, default=14464)
    p.add_argument("--queries", type=int, default=5000)
    a = p.parse_args()
    if a.budget not in (8192, 14464) or a.queries != 5000:
        raise ValueError(
            "Ablation permits B=8192 (Coveo sensitivity) or B=14464 and 5K training queries"
        )
    support, position = collections.Counter(), collections.Counter()
    records = 0
    with a.trace.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["query"]) != records:
                raise ValueError("Training trace query numbering must be 0..4999")
            first = {}
            for pos, raw in enumerate(rec["ids"]):
                vid = int(raw)
                if vid < 0:
                    raise ValueError("Invalid trace vertex ID")
                first.setdefault(vid, pos)
            for vid, pos in first.items():
                if pos > 0:
                    support[vid] += 1
                    position[vid] += pos
            records += 1
    if records != 5000:
        raise ValueError(f"Training trace has {records} queries, not 5000")
    if len(support) < a.budget:
        raise ValueError(
            f"Only {len(support)} eligible vertices, insufficient for matched budget"
        )

    variants = {}
    all_ids = {}
    for mode in MODES:
        ids = select((support, position), mode, a.budget)
        out = a.out / f"landmarks-b{a.budget}-{mode}.bin"
        write_ids(out, ids)
        variants[mode] = {
            "path": str(out),
            "sha256": digest(out),
            "count": len(ids),
            "first20": ids[:20],
            "max_support": max(support[v] for v in ids),
            "median_support": sorted(support[v] for v in ids)[len(ids) // 2],
        }
        all_ids[mode] = set(ids)
    original = a.reference_sum
    if not original.is_file():
        raise FileNotFoundError(original)
    # At the selected Coveo budget all 12,169 eligible vertices are kept.
    # Its 8,192-ID sensitivity experiment must reproduce the FIRST 8,192
    # frozen original sum-ranked IDs, not silently change the reference.
    frozen = original.read_bytes()
    if frozen[:8] != MAGIC or len(frozen) < 16:
        raise ValueError("Invalid frozen landmark file format")
    n, reserved = struct.unpack("<II", frozen[8:16])
    if reserved != 0 or len(frozen) != 16 + 4 * n or n < a.budget:
        raise ValueError("Frozen reference lacks required ranked prefix")
    desired = MAGIC + struct.pack("<II", a.budget, 0) + frozen[16 : 16 + a.budget * 4]
    if (a.out / f"landmarks-b{a.budget}-sum.bin").read_bytes() != desired:
        raise ValueError(
            "The paper's existing sum-position selection could not be reproduced "
            "byte-for-byte from the frozen trace. Do not compare incompatible scores."
        )
    overlaps = {}
    for m in MODES:
        overlaps[m] = {
            "overlap_with_sum": len(all_ids[m] & all_ids["sum"]),
            "fraction": len(all_ids[m] & all_ids["sum"]) / a.budget,
        }
    manifest = {
        "trace": str(a.trace),
        "trace_sha256": digest(a.trace),
        "queries": records,
        "eligible": len(support),
        "budget": a.budget,
        "reference_sum_sha256": digest(original),
        "reference_is_full_selected_core": n == a.budget,
        "ranking_prefix_reproduced": True,
        "configuration": "one shared trace, score-only comparison, fixed B,C,k",
        "definitions": {
            "sum": "sum first expansion positions; ties by support, ID",
            "count": "number of unique query traversals; ties by vertex ID",
            "mean": "mean first expansion position among traversals; ties by support, ID",
        },
        "variants": variants,
        "overlap": overlaps,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("NAVHINTS_SCORE_VARIANTS_REPRODUCED", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
