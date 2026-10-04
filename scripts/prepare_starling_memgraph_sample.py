#!/usr/bin/env python3
"""Prepare tagged samples for Starling's official in-memory Vamana builder.

Modes:
* frequency: choose vertices most often expanded by medoid-start DiskANN on the
  fixed 5000-query training prefix, mirroring Starling's frequency-based option;
* uniform: deterministic uniform sample without replacement.

Outputs are exactly the <prefix>_data.bin and <prefix>_ids.bin files consumed by
Starling tests/build_memory_index. Tags are original PubMed1M vertex IDs.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np


def read_fbin(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim, np.memmap(
        path, dtype="<f4", mode="r", offset=8, shape=(rows, dim)
    )


def write_bin(path: Path, a: np.ndarray) -> None:
    a = np.ascontiguousarray(a)
    if a.ndim != 2:
        raise ValueError("expected 2D array")
    with path.open("wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        a.tofile(f)


def frequency_ids(trace: Path, rows: int, count: int):
    freq = collections.Counter()
    qcount = 0
    for raw in trace.read_text().splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        qcount += 1
        for value in rec["ids"]:
            v = int(value)
            if not 0 <= v < rows:
                raise ValueError(f"trace vertex out of range: {v}")
            freq[v] += 1
    if qcount != 5000:
        raise ValueError(f"expected 5000 training traces, got {qcount}")
    if len(freq) < count:
        raise ValueError(f"only {len(freq)} unique expanded vertices for count {count}")
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    ids = np.asarray([v for v, _ in ranked[:count]], dtype=np.uint32)
    return ids, {
        "training_queries": qcount,
        "unique_expanded_vertices": len(freq),
        "selected_frequency_min": int(ranked[count - 1][1]),
        "selected_frequency_median": float(np.median([x[1] for x in ranked[:count]])),
        "selected_frequency_max": int(ranked[0][1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--mode", choices=("frequency", "uniform"), required=True)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    args.base = args.base.resolve()
    args.out_prefix = args.out_prefix.resolve()
    if args.trace is not None:
        args.trace = args.trace.resolve()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows, dim, x = read_fbin(args.base)
    if not 1 <= args.count <= rows:
        raise ValueError("invalid sample count")

    detail = {}
    if args.mode == "frequency":
        if args.trace is None or not args.trace.is_file():
            raise ValueError("frequency mode requires --trace")
        ids, detail = frequency_ids(args.trace, rows, args.count)
    else:
        rng = np.random.default_rng(args.seed)
        ids = np.asarray(
            rng.choice(rows, size=args.count, replace=False), dtype=np.uint32
        )

    data = np.asarray(x[ids.astype(np.int64)], dtype=np.float32, order="C")
    data_path = Path(str(args.out_prefix) + "_data.bin")
    ids_path = Path(str(args.out_prefix) + "_ids.bin")
    write_bin(data_path, data)
    write_bin(ids_path, ids.reshape(-1, 1).astype("<u4", copy=False))

    manifest = {
        "mode": args.mode,
        "rows": rows,
        "dimension": dim,
        "sample_count": args.count,
        "sampling_fraction": args.count / rows,
        "seed": args.seed if args.mode == "uniform" else None,
        "data_bytes": data_path.stat().st_size,
        "ids_bytes": ids_path.stat().st_size,
        "sample_input_bytes": data_path.stat().st_size + ids_path.stat().st_size,
        "sample_input_mib": (
            data_path.stat().st_size + ids_path.stat().st_size
        ) / (1 << 20),
        "unique_ids": int(len(np.unique(ids))),
        **detail,
    }
    Path(str(args.out_prefix) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
