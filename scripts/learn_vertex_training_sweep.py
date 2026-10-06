#!/usr/bin/env python3
"""Build a nested training-size sweep of vertex continuation maps.

This wrapper intentionally leaves learn_vertex_continuation_maps.py unchanged.
For each requested training-set size it learns the same support-threshold maps on
one deterministic nested sample of the 9K teacher traces, then packs all arms
plus one regional baseline into a single GIVTX001 payload. DiskANN can therefore
compare the whole learning curve using one repacked graph.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

MAGIC = b"GIVTX001"


def parse_payload(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != MAGIC:
        raise ValueError(f"{path}: bad vertex-hint payload")
    nvertices, nvariants, slots, min_position = struct.unpack_from("<IIII", raw, 8)
    off = 24
    values = np.frombuffer(raw, dtype="<u4", count=nvariants, offset=off).copy()
    off += nvariants * 4
    count = nvertices * nvariants * slots
    expected = off + count * 4
    if len(raw) != expected:
        raise ValueError(f"{path}: payload size mismatch")
    data = np.frombuffer(raw, dtype="<u4", count=count, offset=off).reshape(
        nvertices, nvariants, slots
    )
    return (
        int(nvertices),
        int(nvariants),
        int(slots),
        int(min_position),
        values,
        data,
    )


def parse_ints(raw: str):
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--regional", type=Path, required=True)
    ap.add_argument("--learner", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--train-sizes", default="250,500,1000,2000,4000,6000,9000")
    ap.add_argument("--thresholds", default="2,4")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--min-position", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    sizes = parse_ints(args.train_sizes)
    thresholds = parse_ints(args.thresholds)
    if not sizes or sizes[0] <= 0:
        raise ValueError("training sizes must be positive")
    if not thresholds or thresholds[0] <= 0:
        raise ValueError("thresholds must be positive")
    if args.slots <= 0:
        raise ValueError("slots must be positive")

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty teacher trace")
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("teacher trace query IDs must be dense")
    if sizes[-1] > len(records):
        raise ValueError(
            f"largest requested training size {sizes[-1]} exceeds {len(records)} traces"
        )

    # Use one fixed random permutation and nested prefixes so the x-axis changes
    # only the amount of evidence, not the sampling policy.
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    shuffled = [records[i] for i in order]

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    baseline = None
    arms = []
    nvertices = nregions = slots = min_position = None
    regional_source_min_entry_position = None

    for size in sizes:
        prefix = args.work / f"trace-{size}.jsonl"
        with prefix.open("w") as f:
            for qi, src in enumerate(shuffled[:size]):
                rec = dict(src)
                rec["query"] = qi
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

        payload = args.work / f"payload-{size}.bin"
        cmd = [
            sys.executable,
            str(args.learner),
            "--trace", str(prefix),
            "--regional", str(args.regional),
            "--out", str(payload),
            "--thresholds", ",".join(str(x) for x in thresholds),
            "--slots", str(args.slots),
            "--min-position", str(args.min_position),
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        nv, nvar, sl, mp, values, data = parse_payload(payload)
        manifest = json.loads(
            payload.with_suffix(payload.suffix + ".manifest.json").read_text()
        )
        if values.tolist() != [0] + thresholds:
            raise ValueError(f"{payload}: unexpected threshold variants {values.tolist()}")
        if nvar != 1 + len(thresholds):
            raise ValueError(f"{payload}: unexpected variant count")

        if baseline is None:
            nvertices, slots, min_position = nv, sl, mp
            nregions = int(manifest["regions"])
            regional_source_min_entry_position = int(
                manifest["regional_source_min_entry_position"]
            )
            baseline = np.array(data[:, 0, :], copy=True)
        else:
            if (nv, sl, mp) != (nvertices, slots, min_position):
                raise ValueError("inconsistent payload dimensions across training sizes")
            if not np.array_equal(baseline, data[:, 0, :]):
                raise ValueError("regional baseline changed across training sizes")

        for vi, threshold in enumerate(thresholds, start=1):
            coverage = manifest["coverage"][str(threshold)]
            arms.append(
                {
                    "training_queries": size,
                    "min_support": threshold,
                    "coverage": coverage,
                    "support_nonzero_vertices": int(manifest["support_nonzero_vertices"]),
                    "support_nonzero_fraction": float(manifest["support_nonzero_fraction"]),
                    "support_distribution_nonzero": manifest["support_distribution_nonzero"],
                    "pair_updates": int(manifest["pair_updates"]),
                    "data": np.array(data[:, vi, :], copy=True),
                }
            )

    assert baseline is not None
    nvariants = 1 + len(arms)
    packed = np.empty((nvertices, nvariants, slots), dtype="<u4")
    packed[:, 0, :] = baseline

    variant_values = [0]
    variant_manifest = [
        {
            "index": 0,
            "value": 0,
            "name": "regional_full9k",
            "kind": "regional",
            "training_queries": len(records),
            "min_support": None,
        }
    ]

    for idx, arm in enumerate(arms, start=1):
        # Value is descriptive only. Runtime selection is by variant index.
        value = int(arm["training_queries"]) * 100 + int(arm["min_support"])
        if value in variant_values:
            raise ValueError(f"variant value collision: {value}")
        variant_values.append(value)
        packed[:, idx, :] = arm.pop("data")
        variant_manifest.append(
            {
                "index": idx,
                "value": value,
                "name": f"q{arm['training_queries']}_s{arm['min_support']}",
                "kind": "vertex",
                **arm,
            }
        )

    with args.out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", nvertices, nvariants, slots, min_position))
        np.asarray(variant_values, dtype="<u4").tofile(f)
        packed.tofile(f)

    manifest = {
        "format": MAGIC.decode(),
        "purpose": "vertex-continuation learning curve with fixed bootstrap and regional fallback",
        "teacher_traces_available": len(records),
        "sampling": {
            "policy": "one deterministic random permutation; nested prefixes",
            "seed": args.seed,
            "train_sizes": sizes,
        },
        "thresholds": thresholds,
        "vertices": nvertices,
        "regions": nregions,
        "regional_source_min_entry_position": regional_source_min_entry_position,
        "min_training_position": min_position,
        "slots_per_variant": slots,
        "variants_count": nvariants,
        "variant_values": variant_values,
        "variants": variant_manifest,
        "associated_data_bytes": nvariants * slots * 4,
        "learner": str(args.learner),
        "regional": str(args.regional),
        "bytes": args.out.stat().st_size,
        "output": args.out.name,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
