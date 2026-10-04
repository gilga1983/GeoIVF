#!/usr/bin/env python3
"""Build nested ID-only portal pools from the existing 512-region router.

The expensive FP32 router is used only offline as a diversity scaffold. Every
coarse cell contributes its released portal first. Extra portals are actual
members of that cell selected by local spherical k-means and ordered by
farthest-first diversity. A global portal-count budget then adds candidates
round-robin across cells, redistributing unused quota from tiny cells.

Deployment stores only uint32 database IDs. DiskANN scores those IDs with its
already-resident PQ representation. We also emit portal+waypoint pools by
globally deduplicating the existing learned waypoint IDs, testing whether
runtime region conditioning is needed at all.
"""
from __future__ import annotations

import argparse
import heapq
import json
import struct
from pathlib import Path

import faiss
import numpy as np

ROUTER_MAGIC = b"GIPIP001"
CACHE_MAGIC = b"GIWPT001"
START_MAGIC = b"GIDST001"


def read_fbin(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated fbin header")
    rows, dim = struct.unpack("<II", raw)
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size mismatch: {path.stat().st_size} != {expected}")
    x = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    return rows, dim, x


def load_router(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != ROUTER_MAGIC:
        raise ValueError("bad router header")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nfloat = nlist * dim
    off = 16
    centers = np.frombuffer(raw, dtype="<f4", count=nfloat, offset=off).reshape(nlist, dim).copy()
    off += nfloat * 4
    portal_ids = np.frombuffer(raw, dtype="<u4", count=nlist, offset=off).copy()
    off += nlist * 4
    off += nfloat * 4  # released full-precision portal vectors, unused here
    if off != len(raw):
        raise ValueError("router trailing bytes")
    return int(nlist), int(dim), centers, portal_ids


def load_waypoint_ids(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != CACHE_MAGIC:
        raise ValueError("bad waypoint-cache header")
    nlist, total = struct.unpack("<II", raw[8:16])
    off = 16 + (nlist + 1) * 4
    expected = off + total * 4
    if len(raw) != expected:
        raise ValueError("bad waypoint-cache byte length")
    ids = np.frombuffer(raw, dtype="<u4", count=total, offset=off).copy()
    seen = set()
    uniq = []
    for raw_id in ids:
        v = int(raw_id)
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return int(nlist), int(total), np.asarray(uniq, dtype=np.uint32)


def write_start_ids(path: Path, ids):
    arr = np.asarray(ids, dtype="<u4")
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("start ID list must be nonempty 1D")
    if len(np.unique(arr)) != len(arr):
        raise ValueError("duplicate start IDs")
    with path.open("wb") as f:
        f.write(START_MAGIC)
        f.write(struct.pack("<II", len(arr), 0))
        arr.tofile(f)
    expected = 16 + len(arr) * 4
    if path.stat().st_size != expected:
        raise AssertionError("start file byte accounting mismatch")
    return arr


def reservoir_by_cell(x, centers, per_cell: int, seed: int, chunk: int):
    """Deterministic random-priority reservoir for every coarse cell."""
    rows = len(x)
    nlist = len(centers)
    quantizer = faiss.IndexFlatIP(centers.shape[1])
    quantizer.add(np.asarray(centers, dtype=np.float32, order="C"))

    heaps = [[] for _ in range(nlist)]
    counts = np.zeros(nlist, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for start in range(0, rows, chunk):
        block = np.asarray(x[start:start + chunk], dtype=np.float32, order="C")
        _, labels = quantizer.search(block, 1)
        labels = labels[:, 0].astype(np.int32, copy=False)
        priorities = rng.random(len(block), dtype=np.float64)
        counts += np.bincount(labels.astype(np.int64), minlength=nlist)

        order = np.argsort(labels, kind="stable")
        ls = labels[order]
        cuts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1], True])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            cell = int(ls[lo])
            local = order[lo:hi]
            h = heaps[cell]
            for idx in local:
                pr = float(priorities[idx])
                vid = start + int(idx)
                item = (-pr, vid)
                if len(h) < per_cell:
                    heapq.heappush(h, item)
                elif item > h[0]:
                    heapq.heapreplace(h, item)
        print(f"reservoir-assigned={min(start + len(block), rows)}/{rows}", flush=True)

    out = []
    for h in heaps:
        ids = [vid for _, vid in sorted(h, reverse=True)]
        if not ids:
            raise ValueError("empty coarse cell")
        out.append(np.asarray(ids, dtype=np.uint32))
    return out, counts


def local_portals(x, portal_id: int, sample_ids: np.ndarray, max_portals: int, seed: int, cell: int):
    """Released portal first, then as many diverse local representatives as exist."""
    sample_ids = np.asarray(sample_ids, dtype=np.uint32)
    sample = np.asarray(x[sample_ids.astype(np.int64)], dtype=np.float32, order="C")
    k = min(max_portals, len(sample))

    reps = []
    if k == 1:
        reps = [int(sample_ids[0])]
    else:
        km = faiss.Kmeans(
            sample.shape[1], k, niter=10, seed=seed + cell, verbose=False, spherical=True
        )
        km.train(sample)
        centroids = np.asarray(km.centroids, dtype=np.float32, order="C")
        idx = faiss.IndexFlatIP(sample.shape[1])
        idx.add(sample)
        _, nearest = idx.search(centroids, 1)
        reps = [int(sample_ids[i]) for i in nearest[:, 0]]

    candidates = []
    seen = set()
    for v in [int(portal_id), *reps, *map(int, sample_ids)]:
        if v not in seen:
            seen.add(v)
            candidates.append(v)
        if len(candidates) >= min(max_portals, len(sample_ids) + 1):
            break

    if not candidates:
        raise ValueError(f"cell {cell}: no portal candidates")
    if len(candidates) == 1:
        return candidates

    vecs = np.asarray(x[np.asarray(candidates, dtype=np.int64)], dtype=np.float32)
    norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
    norms = np.maximum(norms, 1e-12)
    unit = (vecs / norms[:, None]).astype(np.float32)

    selected = [0]
    remaining = np.ones(len(candidates), dtype=bool)
    remaining[0] = False
    max_sim = unit @ unit[0]
    while np.any(remaining):
        scores = np.where(remaining, max_sim, np.inf)
        nxt = int(np.argmin(scores))
        selected.append(nxt)
        remaining[nxt] = False
        max_sim = np.maximum(max_sim, unit @ unit[nxt])

    return [candidates[i] for i in selected]


def global_nested_pool(ordered, target: int):
    if target < len(ordered):
        raise ValueError("target smaller than one portal per region")
    flat = []
    seen = set()

    # One released portal per coarse region is mandatory.
    for cell in range(len(ordered)):
        v = int(ordered[cell][0])
        if v in seen:
            raise ValueError(f"released portal collision for cell {cell}")
        seen.add(v)
        flat.append(v)

    rank = 1
    max_rank = max(len(v) for v in ordered)
    while len(flat) < target and rank < max_rank:
        for cell in range(len(ordered)):
            if rank >= len(ordered[cell]):
                continue
            v = int(ordered[cell][rank])
            if v in seen:
                continue
            seen.add(v)
            flat.append(v)
            if len(flat) >= target:
                break
        rank += 1

    if len(flat) != target:
        raise ValueError(f"only {len(flat)} unique portals available for target {target}")
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-cell-reservoir", type=int, default=256)
    ap.add_argument("--max-portals-per-region", type=int, default=64)
    ap.add_argument("--counts", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32768)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, dim, x = read_fbin(args.base)
    nlist, rdim, centers, portal_ids = load_router(args.router)
    if dim != rdim:
        raise ValueError("base/router dimension mismatch")
    if nlist != 512:
        raise ValueError(f"expected 512-region router, got {nlist}")
    wnlist, waypoint_pairs, waypoint_unique = load_waypoint_ids(args.waypoint_cache)
    if wnlist != nlist:
        raise ValueError("router/waypoint region mismatch")

    target_counts = [int(v) for v in args.counts.split(",") if v.strip()]
    if min(target_counts) < nlist or target_counts != sorted(target_counts):
        raise ValueError("portal counts must be sorted and >= region count")
    if args.per_cell_reservoir < args.max_portals_per_region:
        raise ValueError("reservoir must be at least max local portals")

    faiss.omp_set_num_threads(args.threads)
    samples, cell_counts = reservoir_by_cell(
        x, centers, args.per_cell_reservoir, args.seed, args.chunk
    )

    ordered = []
    for cell in range(nlist):
        p = local_portals(
            x,
            int(portal_ids[cell]),
            samples[cell],
            args.max_portals_per_region,
            args.seed,
            cell,
        )
        ordered.append(p)
        if (cell + 1) % 32 == 0:
            print(f"local-portals={cell + 1}/{nlist}", flush=True)

    variants = {}
    for target in target_counts:
        flat = global_nested_pool(ordered, target)
        portal_name = f"portals-n{target}"
        portal_path = args.out_dir / f"{portal_name}.bin"
        pids = write_start_ids(portal_path, flat)

        combined = list(map(int, pids))
        combined_seen = set(combined)
        for raw in waypoint_unique:
            v = int(raw)
            if v not in combined_seen:
                combined_seen.add(v)
                combined.append(v)
        combined_name = f"{portal_name}-plus-waypoints"
        combined_path = args.out_dir / f"{combined_name}.bin"
        cids = write_start_ids(combined_path, combined)

        variants[portal_name] = {
            "portal_ids": int(len(pids)),
            "state_bytes": portal_path.stat().st_size,
            "state_mib": portal_path.stat().st_size / (1 << 20),
            "file": portal_path.name,
        }
        variants[combined_name] = {
            "portal_ids": int(len(pids)),
            "waypoint_pairs_source": waypoint_pairs,
            "waypoint_unique_ids": int(len(waypoint_unique)),
            "combined_unique_ids": int(len(cids)),
            "state_bytes": combined_path.stat().st_size,
            "state_mib": combined_path.stat().st_size / (1 << 20),
            "file": combined_path.name,
        }

    local_lengths = np.asarray([len(v) for v in ordered], dtype=np.int64)
    manifest = {
        "dataset_rows": rows,
        "dimension": dim,
        "offline_scaffold": {
            "router_nlist": nlist,
            "router_used_at_runtime": False,
            "selection": (
                "released portal first per coarse cell; deterministic cell reservoir; "
                "local spherical k-means representatives; nested farthest-first diversity; "
                "global portal budgets redistribute unused quota from tiny cells"
            ),
            "database_cell_size": {
                "min": int(cell_counts.min()),
                "median": float(np.median(cell_counts)),
                "max": int(cell_counts.max()),
            },
            "available_local_portals": {
                "min": int(local_lengths.min()),
                "median": float(np.median(local_lengths)),
                "max": int(local_lengths.max()),
            },
        },
        "runtime_representation": "only uint32 database IDs; query scoring reuses DiskANN PQ",
        "waypoint_source": {
            "file": args.waypoint_cache.name,
            "region_vertex_pairs": waypoint_pairs,
            "unique_vertex_ids": int(len(waypoint_unique)),
            "region_conditioning_used_at_runtime": False,
        },
        "variants": variants,
    }
    (args.out_dir / "id-portal-pools.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
