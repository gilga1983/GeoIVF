#!/usr/bin/env python3
"""Reorder Hint-IVF bucket members for PQ-code gather locality.

This is a semantics-preserving layout transformation for GHIVF001 files:
representative IDs and bucket boundaries are unchanged, while child database
IDs inside each bucket are sorted numerically. With canonical fine selection
by (PQ distance, vertex ID), candidate membership and the chosen start are
identical. Only the order in which DiskANN gathers resident PQ-code rows
changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GHIVF001"


def load(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != MAGIC:
        raise ValueError("invalid Hint-IVF header")
    nlist, child_count, total_landmarks, reserved = struct.unpack("<IIII", raw[8:24])
    if nlist == 0 or child_count + nlist != total_landmarks or reserved != 0:
        raise ValueError("invalid Hint-IVF shape")
    expected = 24 + 4 * nlist + 4 * (nlist + 1) + 4 * child_count
    if len(raw) != expected:
        raise ValueError("invalid Hint-IVF byte length")

    off = 24
    medoids = np.frombuffer(raw, dtype="<u4", count=nlist, offset=off).copy()
    off += 4 * nlist
    offsets = np.frombuffer(raw, dtype="<u4", count=nlist + 1, offset=off).copy()
    off += 4 * (nlist + 1)
    children = np.frombuffer(raw, dtype="<u4", count=child_count, offset=off).copy()

    if offsets[0] != 0 or offsets[-1] != child_count or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid Hint-IVF offsets")
    all_ids = np.concatenate((medoids, children))
    if len(np.unique(all_ids)) != total_landmarks:
        raise ValueError("duplicate Hint-IVF IDs")
    return nlist, child_count, total_landmarks, medoids, offsets, children


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    nlist, child_count, total_landmarks, medoids, offsets, children = load(args.input)
    reordered = children.copy()

    displaced = 0
    for cell in range(nlist):
        lo = int(offsets[cell])
        hi = int(offsets[cell + 1])
        before = reordered[lo:hi].copy()
        reordered[lo:hi].sort()
        displaced += int(np.count_nonzero(before != reordered[lo:hi]))

    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", nlist, child_count, total_landmarks, 0))
        medoids.astype("<u4", copy=False).tofile(f)
        offsets.astype("<u4", copy=False).tofile(f)
        reordered.astype("<u4", copy=False).tofile(f)

    # Re-read to fail closed on any serialization mistake.
    n2, c2, t2, m2, o2, ch2 = load(args.output)
    if n2 != nlist or c2 != child_count or t2 != total_landmarks:
        raise AssertionError("shape changed during locality reorder")
    if not np.array_equal(m2, medoids) or not np.array_equal(o2, offsets):
        raise AssertionError("representatives or bucket boundaries changed")
    if sorted(map(int, ch2)) != sorted(map(int, children)):
        raise AssertionError("child candidate set changed")

    manifest = {
        "format": "GHIVF001",
        "transformation": "sort child database IDs ascending within each fixed bucket",
        "semantic_change": False,
        "nlist": int(nlist),
        "child_ids": int(child_count),
        "total_landmarks": int(total_landmarks),
        "state_bytes": int(args.output.stat().st_size),
        "displaced_child_positions": int(displaced),
        "displaced_fraction": float(displaced / child_count) if child_count else 0.0,
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "input": args.input.name,
        "output": args.output.name,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
