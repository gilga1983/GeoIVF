#!/usr/bin/env python3
"""Build the static GeoIVF portal router for PubMed/MedCPT maximum inner product.

The ANN target metric is inner product, so this does not reuse the earlier
normalized-L2 portal construction. Coarse cells use spherical k-means; vectors
are assigned by maximum centroid inner product; each cell's portal is the member
with maximum inner product to its centroid. Query routing probes the highest-IP
coarse cells and chooses the candidate portal with highest query inner product.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import faiss
import numpy as np

MAGIC = b"GIPIP001"


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


def write_router(path: Path, centers, portal_ids, portal_vecs):
    centers = np.asarray(centers, dtype="<f4", order="C")
    portal_ids = np.asarray(portal_ids, dtype="<u4", order="C")
    portal_vecs = np.asarray(portal_vecs, dtype="<f4", order="C")
    if centers.shape != portal_vecs.shape:
        raise ValueError("center/portal shape mismatch")
    if portal_ids.shape != (len(centers),):
        raise ValueError("portal id shape mismatch")
    if not np.isfinite(centers).all() or not np.isfinite(portal_vecs).all():
        raise ValueError("nonfinite router vectors")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", centers.shape[0], centers.shape[1]))
        centers.tofile(f)
        portal_ids.tofile(f)
        portal_vecs.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32768)
    args = ap.parse_args()

    rows, dim, x = read_fbin(args.base)
    if not 1 <= args.nlist <= rows:
        raise ValueError("invalid nlist")
    if args.train_size < args.nlist:
        raise ValueError("training sample smaller than nlist")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    faiss.omp_set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(rows, min(args.train_size, rows), replace=False)
    train = np.asarray(x[train_ids], dtype=np.float32, order="C")

    t0 = time.perf_counter()
    km = faiss.Kmeans(
        dim, args.nlist, niter=20, seed=args.seed, verbose=False, spherical=True
    )
    km.train(train)
    centers = np.asarray(km.centroids, dtype=np.float32, order="C")
    center_norms = np.linalg.norm(centers.astype(np.float64), axis=1)
    if np.max(np.abs(center_norms - 1.0)) > 2e-3:
        raise ValueError("spherical k-means returned non-unit centers")
    train_s = time.perf_counter() - t0

    quantizer = faiss.IndexFlatIP(dim)
    quantizer.add(centers)

    best_scores = np.full(args.nlist, -np.inf, dtype=np.float64)
    portal_ids = np.full(args.nlist, np.iinfo(np.uint32).max, dtype=np.uint32)
    portal_vecs = np.empty((args.nlist, dim), dtype=np.float32)
    counts = np.zeros(args.nlist, dtype=np.int64)

    t0 = time.perf_counter()
    for start in range(0, rows, args.chunk):
        block = np.asarray(x[start:start+args.chunk], dtype=np.float32, order="C")
        scores, labels = quantizer.search(block, 1)
        labels = labels[:, 0].astype(np.int64, copy=False)
        scores = scores[:, 0].astype(np.float64, copy=False)
        counts += np.bincount(labels, minlength=args.nlist)

        # At most 1024 groups per chunk; grouping avoids a Python loop over rows.
        order = np.argsort(labels, kind="stable")
        ls = labels[order]
        ss = scores[order]
        cuts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1], True])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            cell = int(ls[lo])
            rel = int(np.argmax(ss[lo:hi]))
            local = int(order[lo + rel])
            score = float(scores[local])
            if score > best_scores[cell]:
                best_scores[cell] = score
                portal_ids[cell] = np.uint32(start + local)
                portal_vecs[cell] = block[local]
        print(f"assigned={min(start+len(block), rows)}/{rows}", flush=True)

    assign_s = time.perf_counter() - t0
    if np.any(counts == 0):
        raise ValueError(f"{int(np.sum(counts == 0))} empty portal cells")
    if np.any(portal_ids == np.iinfo(np.uint32).max):
        raise ValueError("missing portal")
    if int(portal_ids.max()) >= rows:
        raise ValueError("portal id out of range")

    write_router(args.out, centers, portal_ids, portal_vecs)
    manifest = {
        "dataset": "PubMed MedCPT article embeddings",
        "rows": rows,
        "dimension": dim,
        "metric": "inner_product",
        "coarse_policy": "spherical k-means; assign by max centroid inner product",
        "portal_policy": "cell member with max inner product to centroid",
        "query_policy": "top-nprobe centroids by inner product, then max-IP portal",
        "nlist": args.nlist,
        "training_rows": len(train_ids),
        "seed": args.seed,
        "threads": args.threads,
        "kmeans_seconds": train_s,
        "assignment_seconds": assign_s,
        "cell_count_min": int(counts.min()),
        "cell_count_median": float(np.median(counts)),
        "cell_count_max": int(counts.max()),
        "portal_score_min": float(best_scores.min()),
        "portal_score_median": float(np.median(best_scores)),
        "portal_score_max": float(best_scores.max()),
        "router_bytes": args.out.stat().st_size,
        "router_mib": args.out.stat().st_size / (1 << 20),
        "format_magic": MAGIC.decode(),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
