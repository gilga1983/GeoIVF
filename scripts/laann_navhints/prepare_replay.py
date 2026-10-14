#!/usr/bin/env python3
"""Split native PageANN/LAANN reordered-ID evaluation data causally.

The first 5K queries are DISJOINT junction training. Of the next 5K,
the first 4K causally warm history and only the last 1K are measured.
Do not recompute or reuse old-ID ground truth: LAANN reordered IDs differ.
"""
import argparse
from pathlib import Path
import numpy as np


def load_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype="<u4")
    n, d = int(n), int(d)
    content = np.fromfile(path, dtype=dtype, offset=8)
    if content.size != n * d:
        raise ValueError("Bad binary query/GT length")
    return content.reshape(n, d)


def write_bin(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    with open(path, "wb") as f:
        f.write(np.asarray(array.shape, dtype="<u4").tobytes())
        f.write(array.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--reordered-gt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "uint8", "int8"), default="float32")
    a = p.parse_args()
    typ = {"float32":"<f4", "uint8":"u1", "int8":"i1"}[a.dtype]
    q = load_bin(a.query, typ)
    if len(q) < 10000:
        raise ValueError("At least 10K disjoint query vectors required")
    # Native ground truth can have a float distance array after the ID
    # matrix. Load the IDs only, then emit canonical IDs-only truthset.
    with a.reordered_gt.open("rb") as f:
        h = np.frombuffer(f.read(8), dtype="<u4")
        n, k = map(int, h)
        gt = np.frombuffer(f.read(n * k * 4), dtype="<u4")
    if n < 10000 or gt.size != n * k or k < 10:
        raise ValueError("LAANN-ID ground truth missing/misaligned")
    gt = gt.reshape(n, k)
    if len(q) != n:
        raise ValueError("Query count and translated GT disagree")
    write_bin(a.out / "junction_train.bin", q[:5000])
    write_bin(a.out / "causal_replay.bin", q[5000:10000])
    write_bin(a.out / "causal_replay_native_gt.bin", gt[5000:10000])
    print("NH_LAANN_CAUSAL_SPLIT_PASS: training=5000 warm=4000 measured=1000", a.out)


if __name__ == "__main__":
    main()
