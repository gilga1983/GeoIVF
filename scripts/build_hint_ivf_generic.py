#!/usr/bin/env python3
"""Build an ID-only Hint-IVF for float32/IP or uint8/L2 public datasets.

The persistent deployment file is unchanged GHIVF001: representative database
IDs + CSR bucket offsets + remaining hint IDs. Full vectors are used only
offline to partition the learned vocabulary.

Partition choices:
  spherical  cosine/spherical k-means, then real-vertex medoids
  l2         Euclidean k-means, then nearest real-vertex medoids
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

START_MAGIC = b"GIDST001"
IVF_MAGIC = b"GHIVF001"


def load_ids(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError("bad landmark file")
    n, reserved = struct.unpack("<II", raw[8:16])
    if reserved != 0 or len(raw) != 16 + 4 * n:
        raise ValueError("bad landmark file size")
    ids = np.frombuffer(raw, dtype="<u4", count=n, offset=16).copy()
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("landmark IDs must be unique")
    return ids


def load_xbin(path: Path, data_type: str):
    with path.open("rb") as f:
        raw = f.read(8)
    rows, dim = struct.unpack("<II", raw)
    dtype = {"float32": "<f4", "uint8": "u1"}[data_type]
    itemsize = np.dtype(dtype).itemsize
    expected = 8 + rows * dim * itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"xbin size mismatch {path.stat().st_size} != {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, dim)), rows, dim


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm vector")
    return x / norms


def assign_spherical(x, centers, batch):
    out = np.empty(len(x), dtype=np.int32)
    for lo in range(0, len(x), batch):
        hi = min(len(x), lo + batch)
        out[lo:hi] = np.argmax(x[lo:hi] @ centers.T, axis=1)
    return out


def assign_l2(x, centers, batch):
    out = np.empty(len(x), dtype=np.int32)
    c2 = np.sum(centers * centers, axis=1)
    for lo in range(0, len(x), batch):
        hi = min(len(x), lo + batch)
        xb = x[lo:hi]
        d2 = np.sum(xb * xb, axis=1, keepdims=True) + c2[None, :] - 2.0 * (xb @ centers.T)
        out[lo:hi] = np.argmin(d2, axis=1)
    return out


def lloyd(x, k, iterations, batch, seed, partition):
    rng = np.random.default_rng(seed)
    init = rng.choice(len(x), size=k, replace=False)
    centers = x[init].copy()
    assign_fn = assign_spherical if partition == "spherical" else assign_l2

    for _ in range(iterations):
        assign = assign_fn(x, centers, batch)
        new_centers = np.empty_like(centers)
        for c in range(k):
            members = x[assign == c]
            if len(members) == 0:
                new_centers[c] = x[rng.integers(0, len(x))]
            else:
                v = members.mean(axis=0)
                if partition == "spherical":
                    n = np.linalg.norm(v)
                    new_centers[c] = v / n if n > 0 else x[rng.integers(0, len(x))]
                else:
                    new_centers[c] = v
        centers = new_centers
    return centers, assign_fn(x, centers, batch)


def choose_medoids(x, centers, assign, partition):
    out = np.empty(len(centers), dtype=np.int32)
    for c in range(len(centers)):
        members = np.flatnonzero(assign == c)
        if len(members) == 0:
            raise ValueError(f"empty cluster {c}")
        xm = x[members]
        if partition == "spherical":
            score = xm @ centers[c]
            out[c] = members[int(np.argmax(score))]
        else:
            delta = xm - centers[c]
            d2 = np.sum(delta * delta, axis=1)
            out[c] = members[int(np.argmin(d2))]
    if len(np.unique(out)) != len(out):
        raise ValueError("duplicate medoids")
    return out


def write_ivf(path, medoid_ids, buckets, total):
    medoid_set = set(map(int, medoid_ids))
    seen = set(medoid_set)
    offsets = [0]
    children = []
    for bucket in buckets:
        for raw in bucket:
            vid = int(raw)
            if vid in medoid_set:
                continue
            if vid in seen:
                raise ValueError(f"duplicate child {vid}")
            seen.add(vid)
            children.append(vid)
        offsets.append(len(children))
    if len(seen) != total:
        raise ValueError(f"IVF covers {len(seen)} IDs, expected {total}")

    medoid_ids = np.asarray(medoid_ids, dtype="<u4")
    offsets = np.asarray(offsets, dtype="<u4")
    children = np.asarray(children, dtype="<u4")
    with path.open("wb") as f:
        f.write(IVF_MAGIC)
        f.write(struct.pack("<IIII", len(medoid_ids), len(children), total, 0))
        medoid_ids.tofile(f)
        offsets.tofile(f)
        children.tofile(f)
    return offsets, children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-data-type", choices=("float32", "uint8"), required=True)
    ap.add_argument("--hints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--partition", choices=("spherical", "l2"), required=True)
    ap.add_argument("--nlist", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hint_ids = load_ids(args.hints)
    if not 1 <= args.nlist <= len(hint_ids):
        raise ValueError("bad nlist")
    base, nbase, dim = load_xbin(args.base, args.base_data_type)
    if int(hint_ids.max()) >= nbase:
        raise ValueError("hint ID outside base")

    x_raw = np.asarray(base[hint_ids.astype(np.int64)], dtype=np.float32)
    raw_norms = np.linalg.norm(x_raw, axis=1)
    x = normalize(x_raw) if args.partition == "spherical" else x_raw

    centers, initial = lloyd(
        x, args.nlist, args.iterations, args.batch, args.seed, args.partition
    )
    medoid_local = choose_medoids(x, centers, initial, args.partition)
    medoid_vecs = x[medoid_local]
    assign_fn = assign_spherical if args.partition == "spherical" else assign_l2
    final = assign_fn(x, medoid_vecs, args.batch)
    medoid_ids = hint_ids[medoid_local]
    buckets = [hint_ids[final == c] for c in range(args.nlist)]
    offsets, children = write_ivf(args.out, medoid_ids, buckets, len(hint_ids))

    sizes = np.diff(offsets.astype(np.int64))
    result = {
        "format": "GHIVF001",
        "partition": args.partition,
        "base_data_type": args.base_data_type,
        "base_rows": nbase,
        "dimension": dim,
        "landmark_ids": len(hint_ids),
        "nlist": args.nlist,
        "child_ids": len(children),
        "state_bytes": args.out.stat().st_size,
        "bucket_children_min": int(sizes.min()),
        "bucket_children_median": float(np.median(sizes)),
        "bucket_children_mean": float(sizes.mean()),
        "bucket_children_p95": float(np.quantile(sizes, 0.95)),
        "bucket_children_max": int(sizes.max()),
        "hint_norm_min": float(raw_norms.min()),
        "hint_norm_median": float(np.median(raw_norms)),
        "hint_norm_max": float(raw_norms.max()),
        "iterations": args.iterations,
        "seed": args.seed,
        "file": args.out.name,
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
