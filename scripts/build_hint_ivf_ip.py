#!/usr/bin/env python3
"""Build an ID-only inner-product IVF over learned navigation hints.

Deployment stores only database IDs and CSR offsets. Offline construction uses
the full hint vectors to partition them around *actual hint vertices* under raw
inner product. Runtime geometry still comes exclusively from DiskANN's resident
PQ representation.

Each Lloyd-style round:
  1) assign every hint to the medoid with maximum inner product;
  2) for each bucket, choose the member maximizing inner product with the
     bucket sum, i.e. maximizing aggregate similarity to members.

Thus the stored representatives are always real database IDs and the partition
matches the same inner-product objective used by the target DiskANN index.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

START_MAGIC = b"GIDST001"
IVF_MAGIC = b"GHIVF001"


def load_start_ids(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != START_MAGIC:
        raise ValueError("bad global-start file")
    count, reserved = struct.unpack("<II", raw[8:16])
    if reserved != 0 or len(raw) != 16 + count * 4:
        raise ValueError("bad global-start file size")
    ids = np.frombuffer(raw, dtype="<u4", count=count, offset=16).copy()
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("start IDs must be nonempty and unique")
    return ids


def load_fbin_memmap(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"bad fbin size: got {path.stat().st_size}, expected {expected}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim)), rows, dim


def assign_batched(x: np.ndarray, medoids: np.ndarray, batch: int):
    assign = np.empty(len(x), dtype=np.int32)
    best = np.empty(len(x), dtype=np.float32)
    for lo in range(0, len(x), batch):
        hi = min(len(x), lo + batch)
        sims = x[lo:hi] @ medoids.T
        idx = np.argmax(sims, axis=1)
        assign[lo:hi] = idx
        best[lo:hi] = sims[np.arange(hi - lo), idx]
    return assign, best


def reseed_empty(
    x: np.ndarray,
    medoid_local: np.ndarray,
    assign: np.ndarray,
    best: np.ndarray,
) -> np.ndarray:
    """Fill empty clusters with poorly represented non-medoid points."""
    counts = np.bincount(assign, minlength=len(medoid_local))
    empty = np.flatnonzero(counts == 0)
    if len(empty) == 0:
        return medoid_local

    chosen = set(map(int, medoid_local))
    order = np.argsort(best)  # low maximum IP = poorly represented
    cursor = 0
    out = medoid_local.copy()
    for c in empty:
        while cursor < len(order) and int(order[cursor]) in chosen:
            cursor += 1
        if cursor == len(order):
            raise RuntimeError("unable to reseed empty cluster uniquely")
        idx = int(order[cursor])
        cursor += 1
        chosen.add(idx)
        out[c] = idx
    return out


def update_medoids_ip(x: np.ndarray, assign: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(k, dtype=np.int32)
    for c in range(k):
        members = np.flatnonzero(assign == c)
        if len(members) == 0:
            out[c] = -1
            continue
        # argmax_j x_j^T sum_i x_i maximizes aggregate within-bucket IP.
        bucket_sum = x[members].sum(axis=0, dtype=np.float64).astype(np.float32)
        scores = x[members] @ bucket_sum
        out[c] = int(members[int(np.argmax(scores))])
    return out


def ip_medoids(
    x: np.ndarray,
    k: int,
    iterations: int,
    batch: int,
    seed: int,
):
    if not 1 <= k <= len(x):
        raise ValueError("invalid number of IVF lists")
    rng = np.random.default_rng(seed)
    medoid_local = rng.choice(len(x), size=k, replace=False).astype(np.int32)

    history = []
    for iteration in range(iterations):
        medoid_vecs = x[medoid_local]
        assign, best = assign_batched(x, medoid_vecs, batch)
        medoid_local = reseed_empty(x, medoid_local, assign, best)
        # Reassign if reseeding changed any representative.
        assign, best = assign_batched(x, x[medoid_local], batch)

        updated = update_medoids_ip(x, assign, k)
        missing = np.flatnonzero(updated < 0)
        if len(missing):
            # Defensive only; assign after reseeding should leave no empties.
            updated = reseed_empty(x, medoid_local, assign, best)

        changed = int(np.sum(updated != medoid_local))
        history.append(
            {
                "iteration": iteration,
                "changed_medoids": changed,
                "mean_best_ip": float(best.mean()),
                "median_best_ip": float(np.median(best)),
            }
        )
        medoid_local = updated
        if changed == 0:
            break

    final_assign, final_best = assign_batched(x, x[medoid_local], batch)
    # A last defensive empty check.
    counts = np.bincount(final_assign, minlength=k)
    if np.any(counts == 0):
        medoid_local = reseed_empty(x, medoid_local, final_assign, final_best)
        final_assign, final_best = assign_batched(x, x[medoid_local], batch)
        counts = np.bincount(final_assign, minlength=k)
        if np.any(counts == 0):
            raise RuntimeError("empty cluster remains after reseed")

    if len(np.unique(medoid_local)) != k:
        raise RuntimeError("duplicate medoids after IP clustering")
    return medoid_local, final_assign, final_best, history


def write_ivf(path: Path, medoid_ids: np.ndarray, buckets: list[np.ndarray], total_landmarks: int):
    nlist = len(medoid_ids)
    medoid_set = set(map(int, medoid_ids))
    seen = set(medoid_set)
    offsets = [0]
    children: list[int] = []

    for bucket in buckets:
        for raw in bucket:
            vid = int(raw)
            if vid in medoid_set:
                continue
            if vid in seen:
                raise ValueError(f"duplicate child ID {vid}")
            seen.add(vid)
            children.append(vid)
        offsets.append(len(children))

    if len(seen) != total_landmarks:
        raise ValueError(f"IVF covers {len(seen)} IDs, expected {total_landmarks}")

    medoid_ids = np.asarray(medoid_ids, dtype="<u4")
    offsets_arr = np.asarray(offsets, dtype="<u4")
    children_arr = np.asarray(children, dtype="<u4")
    with path.open("wb") as f:
        f.write(IVF_MAGIC)
        f.write(struct.pack("<IIII", nlist, len(children_arr), total_landmarks, 0))
        medoid_ids.tofile(f)
        offsets_arr.tofile(f)
        children_arr.tofile(f)
    return offsets_arr, children_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--hints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nlist", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    args.base = args.base.resolve()
    args.hints = args.hints.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hint_ids = load_start_ids(args.hints)
    base, nbase, dim = load_fbin_memmap(args.base)
    if int(hint_ids.max()) >= nbase:
        raise ValueError("hint ID outside base dataset")

    # Copy only the hint vectors into RAM. For 16K x 768 f32 this is ~47 MiB.
    x = np.asarray(base[hint_ids.astype(np.int64)], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)

    medoid_local, assign, best_ip, history = ip_medoids(
        x, args.nlist, args.iterations, args.batch, args.seed
    )
    medoid_ids = hint_ids[medoid_local]
    buckets = [hint_ids[assign == c] for c in range(args.nlist)]
    offsets, children = write_ivf(args.out, medoid_ids, buckets, len(hint_ids))

    sizes = np.diff(offsets.astype(np.int64))
    manifest = {
        "format": "GHIVF001",
        "partition": "raw-inner-product k-medoids over actual hint vertices",
        "base_rows": int(nbase),
        "dimension": int(dim),
        "landmark_ids": int(len(hint_ids)),
        "nlist": int(args.nlist),
        "medoid_ids": int(len(medoid_ids)),
        "child_ids": int(len(children)),
        "state_bytes": int(args.out.stat().st_size),
        "bucket_children_min": int(sizes.min()),
        "bucket_children_median": float(np.median(sizes)),
        "bucket_children_mean": float(sizes.mean()),
        "bucket_children_p95": float(np.quantile(sizes, 0.95)),
        "bucket_children_max": int(sizes.max()),
        "hint_norm_min": float(norms.min()),
        "hint_norm_median": float(np.median(norms)),
        "hint_norm_max": float(norms.max()),
        "mean_assignment_ip": float(best_ip.mean()),
        "median_assignment_ip": float(np.median(best_ip)),
        "iterations_requested": int(args.iterations),
        "iterations_completed": int(len(history)),
        "history": history,
        "seed": int(args.seed),
        "file": args.out.name,
    }
    mp = args.out.with_suffix(args.out.suffix + ".manifest.json")
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
