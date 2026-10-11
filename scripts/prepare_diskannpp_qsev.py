#!/usr/bin/env python3
"""Build a memory-budgeted DiskANN++-style query-sensitive entry pool.

DiskANN++ QSEV clusters the database, queries the graph once per centroid to
obtain representative graph vertices, adds the static central vertex, and then
linearly scans those candidate entry vectors for every incoming query.

For a conservative same-memory baseline, this builder strengthens the offline
step: it uses the exact nearest database vector to every centroid rather than
an approximate graph search. The deployed state and online rule remain the
QSEV rule: a flat list of full-precision entry vectors and IDs scanned per query.

The default 1023 clusters plus the DiskANN medoid produce exactly 1024 stored
entries. At 768 dimensions this is 3,149,840 bytes including the file header
and IDs, just below the current 512-region learned-navigation budget.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import faiss
import numpy as np

MAGIC = b"GQSEV001"


def read_fbin(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size mismatch: {path.stat().st_size} != {expected}")
    data = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    return rows, dim, data


def locate_disk_index(prefix: Path) -> Path:
    candidates = [
        p for p in prefix.parent.glob(prefix.name + "*")
        if p.is_file() and ("disk.index" in p.name or p.name.endswith("_disk.index"))
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one disk-index file for {prefix}, got {candidates}")
    return candidates[0]


def read_medoid(disk_index: Path, expected_rows: int, expected_dim: int) -> int:
    # Rust DiskANN reads GraphHeader from sector bytes [8..]. GraphMetadata is
    # ten little-endian u64 values: num_pts, dims, medoid, ...
    with disk_index.open("rb") as f:
        raw = f.read(8 + 80)
    if len(raw) < 32:
        raise ValueError("truncated DiskANN graph header")
    num_pts, dims, medoid = struct.unpack_from("<QQQ", raw, 8)
    if num_pts != expected_rows or dims != expected_dim:
        raise ValueError(
            f"DiskANN graph header mismatch: {num_pts}x{dims}, "
            f"expected {expected_rows}x{expected_dim}"
        )
    if medoid >= expected_rows:
        raise ValueError("DiskANN medoid out of range")
    return int(medoid)


def write_pool(path: Path, ids: np.ndarray, vecs: np.ndarray) -> None:
    ids = np.asarray(ids, dtype="<u4", order="C")
    vecs = np.asarray(vecs, dtype="<f4", order="C")
    if vecs.ndim != 2 or ids.shape != (len(vecs),):
        raise ValueError("QSEV pool shape mismatch")
    if not np.isfinite(vecs).all():
        raise ValueError("nonfinite QSEV vector")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(ids), vecs.shape[1]))
        ids.tofile(f)
        vecs.tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clusters", type=int, default=1023)
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    args.base = args.base.resolve()
    args.index_prefix = args.index_prefix.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, dim, x = read_fbin(args.base)
    if args.clusters < 1 or args.clusters + 1 > rows:
        raise ValueError("invalid cluster count")
    if args.train_size < args.clusters:
        raise ValueError("training sample smaller than cluster count")

    faiss.omp_set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(rows, min(args.train_size, rows), replace=False)
    train = np.asarray(x[train_ids], dtype=np.float32, order="C")

    print(
        f"training {args.clusters} spherical centroids on {len(train)} rows",
        flush=True,
    )
    t0 = time.perf_counter()
    km = faiss.Kmeans(
        dim,
        args.clusters,
        niter=20,
        seed=args.seed,
        verbose=False,
        spherical=True,
    )
    km.train(train)
    centers = np.asarray(km.centroids, dtype=np.float32, order="C")
    train_seconds = time.perf_counter() - t0
    del train

    # QSEV uses ANNS to map each centroid to a graph vertex. Exact top-1 makes
    # this baseline deliberately stronger while preserving the same deployed
    # candidate-list representation and online selection rule.
    print("finding exact database representative for each centroid", flush=True)
    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    chunk = 32768
    for start in range(0, rows, chunk):
        flat.add(np.asarray(x[start : start + chunk], dtype=np.float32, order="C"))
        if start % (chunk * 8) == 0:
            print(f"exact-index rows={min(start + chunk, rows)}/{rows}", flush=True)
    _, centroid_ids = flat.search(centers, 1)
    exact_seconds = time.perf_counter() - t0
    centroid_ids = centroid_ids[:, 0].astype(np.uint32, copy=False)
    if np.any(centroid_ids >= rows):
        raise ValueError("exact representative out of range")

    disk_index = locate_disk_index(args.index_prefix)
    medoid = read_medoid(disk_index, rows, dim)
    ids = np.concatenate(
        [np.asarray([medoid], dtype=np.uint32), centroid_ids],
    ).astype(np.uint32, copy=False)
    vecs = np.asarray(x[ids.astype(np.int64)], dtype=np.float32, order="C")
    write_pool(args.out, ids, vecs)

    expected_bytes = 16 + len(ids) * 4 + len(ids) * dim * 4
    if args.out.stat().st_size != expected_bytes:
        raise AssertionError("QSEV file byte accounting mismatch")

    manifest = {
        "baseline": "DiskANN++ query-sensitive entry vertex (QSEV) component",
        "offline_strengthening": (
            "exact database nearest vertex to each centroid, rather than "
            "the paper's approximate Vamana search"
        ),
        "online_policy": "linear scan of stored full-precision entry vectors",
        "metric": "inner_product",
        "rows": rows,
        "dimension": dim,
        "clusters": args.clusters,
        "stored_entries": len(ids),
        "diskann_medoid": medoid,
        "training_rows": len(train_ids),
        "seed": args.seed,
        "threads": args.threads,
        "kmeans_seconds": train_seconds,
        "exact_representative_seconds": exact_seconds,
        "unique_entry_ids": int(len(np.unique(ids))),
        "entry_file_bytes": args.out.stat().st_size,
        "entry_file_mib": args.out.stat().st_size / (1 << 20),
        "vector_payload_bytes": int(len(ids) * dim * 4),
        "id_payload_bytes": int(len(ids) * 4),
        "format_magic": MAGIC.decode(),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
