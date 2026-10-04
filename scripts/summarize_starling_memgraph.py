#!/usr/bin/env python3
"""Summarize an official Starling MemGraph build and navigation sweep."""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


def bin_shape(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated bin header: {path}")
    rows, cols = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * cols * 4:
        raise ValueError(f"unexpected uint32 result size: {path}")
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("frequency", "uniform"), required=True)
    ap.add_argument("--sample-count", type=int, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--search-log", type=Path, required=True)
    ap.add_argument("--result-prefix", type=Path, required=True)
    ap.add_argument("--search-ls", default="1,2,4,8,16,32,64")
    ap.add_argument("--target-bytes", type=int, default=3_159_860)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.index_prefix = args.index_prefix.resolve()
    args.search_log = args.search_log.resolve()
    args.result_prefix = args.result_prefix.resolve()
    args.out = args.out.resolve()

    ls = [int(x) for x in args.search_ls.split(",") if x.strip()]
    index_files = sorted(
        p for p in args.index_prefix.parent.glob(args.index_prefix.name + "*")
        if p.is_file()
    )
    if not index_files:
        raise FileNotFoundError(f"no Starling index files for {args.index_prefix}")
    index_bytes = sum(p.stat().st_size for p in index_files)

    rows = {}
    for raw in args.search_log.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        l = int(parts[0])
        if l not in ls:
            continue
        try:
            qps = float(parts[1])
            mean_us = float(parts[2])
            p999_us = float(parts[3])
        except ValueError:
            continue
        rows[l] = {
            "nav_qps": qps,
            "nav_mean_latency_us": mean_us,
            "nav_p999_latency_us": p999_us,
        }

    missing = [l for l in ls if l not in rows]
    if missing:
        raise ValueError(f"missing Starling search rows for L={missing}")

    results = {}
    for l in ls:
        result = Path(str(args.result_prefix) + f"_{l}_idx_uint32.bin")
        if not result.is_file():
            raise FileNotFoundError(result)
        n, cols = bin_shape(result)
        if (n, cols) != (5000, 1):
            raise ValueError(f"{result}: expected 5000x1, got {n}x{cols}")
        results[str(l)] = {
            **rows[l],
            "start_file": str(result),
            "start_file_bytes": result.stat().st_size,
        }

    obj = {
        "implementation": "official zilliztech/starling in-memory Vamana",
        "mode": args.mode,
        "sample_count": args.sample_count,
        "sample_fraction": args.sample_count / 1_000_000,
        "metric": "mips",
        "R": 48,
        "Lbuild": 128,
        "alpha": 1.2,
        "index_files": [str(p) for p in index_files],
        "deployed_index_bytes_on_disk": index_bytes,
        "deployed_index_mib_on_disk": index_bytes / (1 << 20),
        "target_bytes": args.target_bytes,
        "budget_fraction": index_bytes / args.target_bytes,
        "budget_guard": (
            "saved-index bytes are used as a conservative lower bound on runtime "
            "RAM; allocator/object overhead is not credited to Starling"
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
