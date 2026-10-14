#!/usr/bin/env python3
"""Train native-LAANN page-junctions and 512 PQ-geometric routing regions.

Requires a frozen 5,000-query LAANN page-expansion trace and the ORIGINAL
native LAANN reordered PQ codes/pivots, which are read but never modified.
All IDs are native PageANN/LAANN reordered vector IDs.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_bin_blob(path, offset, dtype):
    with open(path, "rb") as f:
        f.seek(int(offset))
        shape = np.frombuffer(f.read(8), dtype="<u4")
        if len(shape) != 2 or not int(shape[0]) or not int(shape[1]):
            raise ValueError(f"Bad diskann bin shape at offset {offset}")
        data = np.frombuffer(f.read(int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize),
                             dtype=dtype)
    if data.size != int(shape[0]) * int(shape[1]):
        raise ValueError("Truncated DiskANN binary blob")
    return data.reshape(tuple(map(int, shape)))


def read_pivots(p):
    with p.open("rb") as f:
        shape = np.frombuffer(f.read(8), dtype="<u4")
        if tuple(shape) not in ((4, 1), (5, 1)):
            raise ValueError(f"Unsupported LAANN pivot offsets shape: {shape}")
        offsets = np.frombuffer(f.read(int(shape[0]) * 8), dtype="<u8")
    table = read_bin_blob(p, int(offsets[0]), "<f4")
    if table.shape[0] != 256:
        raise ValueError(f"Expected 256 codebook pivots, got {table.shape}")
    offset_index = 2 if len(offsets) == 4 else 3
    chunks = read_bin_blob(p, int(offsets[offset_index]), "<u4").ravel()
    if chunks[0] != 0 or chunks[-1] != table.shape[1]:
        raise ValueError("PQ chunk offsets do not match codebook dimensions")
    return table, chunks


def read_pq(p, num_chunks):
    with p.open("rb") as f:
        header = np.frombuffer(f.read(8), dtype="<u4")
    if len(header) != 2 or int(header[1]) != num_chunks:
        raise ValueError("Native LAANN reordered PQ chunk count mismatch")
    n = int(header[0])
    if p.stat().st_size != 8 + n * num_chunks:
        raise ValueError("Truncated or extraneous native PQ codes")
    return np.memmap(p, mode="r", dtype="u1", offset=8, shape=(n, num_chunks))


def decode(ids, pq, table, bounds):
    if np.any(ids >= pq.shape[0]) or np.any(ids < 0):
        raise ValueError("Trained junction ID outside reordered index")
    codes = np.asarray(pq[ids], dtype="u1")
    data = np.empty((len(ids), table.shape[1]), dtype=np.float32)
    for c in range(codes.shape[1]):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        data[:, lo:hi] = table[codes[:, c], lo:hi]
    return data


def clustering(x, requested, seed, iterations):
    """Deterministic geometric grouping in the native PQ reconstruction space."""
    n = len(x)
    regions = min(requested, n)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=regions, replace=False)].copy()
    norms = np.sum(x * x, axis=1)
    assignments = None
    for _ in range(iterations):
        dist = norms[:, None] + np.sum(centers * centers, axis=1)[None, :] - 2 * (x @ centers.T)
        assignments = np.argmin(dist, axis=1)
        counts = np.bincount(assignments, minlength=regions)
        # Do not silently reduce the number of geometric regions.
        for empty in np.where(counts == 0)[0]:
            donor = int(np.argmax(counts))
            candidates = np.where(assignments == donor)[0]
            far = int(candidates[np.argmax(dist[candidates, donor])])
            assignments[far] = empty
            counts[donor] -= 1
            counts[empty] += 1
        new_centers = np.zeros_like(centers)
        np.add.at(new_centers, assignments, x)
        new_centers /= counts[:, None]
        centers = new_centers
    return assignments, centers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--reordered-pq", type=Path, required=True)
    p.add_argument("--pivots", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-queries", type=int, default=5000)
    p.add_argument("--budget", type=int, default=14464)
    p.add_argument("--regions", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261014)
    a = p.parse_args()
    if a.budget != 14464 or a.regions != 512:
        raise ValueError("Paper integration requires B=14464 and 512 regions")
    if a.training_queries <= 0:
        raise ValueError("Invalid training query count")

    support, position = collections.Counter(), collections.Counter()
    trace_queries = 0
    for line in a.trace.read_text().splitlines():
        trace_queries += 1
        ids = [int(v) for v in line.split()]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated page ID within query visit trace")
        for rank, vid in enumerate(ids, 1):
            support[vid] += 1
            position[vid] += rank
    if trace_queries != a.training_queries:
        raise ValueError(f"Expected {a.training_queries} queries; got {trace_queries}")
    if len(support) < a.regions:
        raise ValueError("Not enough distinct visited page representatives for 512 regions")
    ids = np.array(sorted(support, key=lambda v: (-position[v], -support[v], v))[:a.budget],
                   dtype=np.int64)

    pivots, bounds = read_pivots(a.pivots)
    pq = read_pq(a.reordered_pq, len(bounds) - 1)
    vectors = decode(ids, pq, pivots, bounds)
    assignment, centers = clustering(vectors, a.regions, a.seed, iterations=5)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as f:
        f.write(f"NHLAANN1 {a.regions}\n")
        for region in range(a.regions):
            inds = np.where(assignment == region)[0]
            if not len(inds):
                raise ValueError("Empty geometric region")
            d = np.sum((vectors[inds] - centers[region]) ** 2, axis=1)
            rep = int(ids[inds[int(np.argmin(d))]])
            members = [int(v) for v in ids[inds]]
            f.write(f"{rep} {len(members)} " + " ".join(map(str, members)) + "\n")

    manifest = {
        "design": "LAANN native page-visit junctions + PQ-space geometric directory",
        "junctions": len(ids), "regions": a.regions, "query_prefix": trace_queries,
        "reordered_vector_ids": True, "page_representative_training": True,
        "score": "sum first-expansion positions across training traversals",
        "seed": a.seed, "iterations": 5,
        "source_sha256": {"trace": sha(a.trace), "reordered_pq": sha(a.reordered_pq),
                          "pivots": sha(a.pivots)},
        "directory_sha256": sha(a.output)
    }
    a.output.with_suffix(a.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("NH_LAANN_DIRECTORY_PASS", json.dumps(manifest))


if __name__ == "__main__":
    main()
