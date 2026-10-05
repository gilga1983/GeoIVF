#!/usr/bin/env python3
"""Build an ID-only IVF over learned navigation hints.

The deployment structure stores no vectors or PQ codes. It contains only:
  * nlist representative database IDs,
  * CSR bucket offsets, and
  * the remaining learned hint IDs.

Offline clustering uses full base vectors only to decide the partition.
Runtime routing scores representatives and selected bucket members using the
PQ representation already resident in DiskANN.
"""
from __future__ import annotations

import argparse
import hashlib
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
    if reserved != 0 or len(raw) != 16 + 4 * count:
        raise ValueError("bad global-start file size")
    ids = np.frombuffer(raw, dtype="<u4", count=count, offset=16).copy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("global-start IDs are not unique")
    return ids


def load_fbin_memmap(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"bad fbin size: got {path.stat().st_size}, expected {expected}")
    mm = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    return mm, rows, dim


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm hint vector")
    return x / norms


def assign_batched(x: np.ndarray, centers: np.ndarray, batch: int) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int32)
    for lo in range(0, x.shape[0], batch):
        hi = min(x.shape[0], lo + batch)
        sims = x[lo:hi] @ centers.T
        out[lo:hi] = np.argmax(sims, axis=1)
    return out


def lloyd_spherical(
    x: np.ndarray,
    k: int,
    iterations: int,
    batch: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= len(x):
        raise ValueError("invalid number of IVF lists")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if batch <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(seed)
    init = rng.choice(len(x), size=k, replace=False)
    centers = x[init].copy()

    for _ in range(iterations):
        assign = assign_batched(x, centers, batch)
        new_centers = np.empty_like(centers)
        for c in range(k):
            members = x[assign == c]
            if len(members) == 0:
                new_centers[c] = x[rng.integers(0, len(x))]
            else:
                v = members.mean(axis=0)
                n = np.linalg.norm(v)
                new_centers[c] = v / n if n > 0 else x[rng.integers(0, len(x))]
        centers = new_centers

    assign = assign_batched(x, centers, batch)
    return centers, assign


def choose_medoids(x: np.ndarray, centers: np.ndarray, assign: np.ndarray) -> np.ndarray:
    medoid_local = np.empty(len(centers), dtype=np.int32)
    for c in range(len(centers)):
        members = np.flatnonzero(assign == c)
        if len(members) == 0:
            raise ValueError(f"empty cluster {c}")
        sims = x[members] @ centers[c]
        medoid_local[c] = members[int(np.argmax(sims))]
    if len(np.unique(medoid_local)) != len(medoid_local):
        raise ValueError("duplicate medoids")
    return medoid_local


def write_ivf(path: Path, medoid_ids: np.ndarray, buckets: list[np.ndarray], total_landmarks: int):
    nlist = len(medoid_ids)
    offsets = [0]
    children = []
    medoid_set = set(map(int, medoid_ids))
    seen = set(medoid_set)

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
    offsets = np.asarray(offsets, dtype="<u4")
    children = np.asarray(children, dtype="<u4")
    with path.open("wb") as f:
        f.write(IVF_MAGIC)
        f.write(struct.pack("<IIII", nlist, len(children), total_landmarks, 0))
        medoid_ids.tofile(f)
        offsets.tofile(f)
        children.tofile(f)
    return offsets, children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--hints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    args.base = args.base.resolve()
    args.hints = args.hints.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.nlist <= 0:
        raise ValueError("--nlist must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")

    hint_ids = load_start_ids(args.hints)
    if args.nlist > len(hint_ids):
        raise ValueError(
            f"--nlist={args.nlist} exceeds {len(hint_ids)} learned hints"
        )

    base, nbase, dim = load_fbin_memmap(args.base)
    if int(hint_ids.max()) >= nbase:
        raise ValueError("hint ID outside base dataset")

    x_raw = np.asarray(base[hint_ids.astype(np.int64)], dtype=np.float32)
    raw_norms = np.linalg.norm(x_raw, axis=1)
    x = normalize_rows(x_raw)

    centers, initial_assign = lloyd_spherical(
        x, args.nlist, args.iterations, args.batch, args.seed
    )
    medoid_local = choose_medoids(x, centers, initial_assign)

    # The runtime router scores real medoid vertices, not floating centroids.
    # Reassign every hint to the actual medoid it will see at deployment.
    medoid_vecs = x[medoid_local]
    final_assign = assign_batched(x, medoid_vecs, args.batch)
    medoid_ids = hint_ids[medoid_local]

    buckets = [hint_ids[final_assign == c] for c in range(args.nlist)]
    offsets, children = write_ivf(args.out, medoid_ids, buckets, len(hint_ids))

    sizes = np.diff(offsets.astype(np.int64))
    manifest = {
        "format": "GHIVF001",
        "builder": "ID-only spherical k-means with real database medoids",
        "metric_used_for_partition": "cosine/spherical offline; runtime uses DiskANN resident PQ metric",
        "source_hints_file": args.hints.name,
        "source_hints_sha256": sha256_file(args.hints),
        "base_rows": nbase,
        "dimension": dim,
        "landmark_ids": int(len(hint_ids)),
        "nlist": int(args.nlist),
        "medoid_ids": int(len(medoid_ids)),
        "child_ids": int(len(children)),
        "state_bytes": args.out.stat().st_size,
        "bucket_children_min": int(sizes.min()),
        "bucket_children_median": float(np.median(sizes)),
        "bucket_children_mean": float(sizes.mean()),
        "bucket_children_p95": float(np.quantile(sizes, 0.95)),
        "bucket_children_max": int(sizes.max()),
        "base_vector_norm_min": float(raw_norms.min()),
        "base_vector_norm_median": float(np.median(raw_norms)),
        "base_vector_norm_max": float(raw_norms.max()),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "file": args.out.name,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
