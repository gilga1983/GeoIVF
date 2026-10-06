#!/usr/bin/env python3
"""Screen DiskANN PQ codes as certified semantic filters on held-out searches.

For each database vector x with PQ reconstruction x_hat, compute once:

    epsilon_x = ||x - x_hat||_2.

For query q this gives the certified inner-product interval

    q.x in [q.x_hat - ||q|| epsilon_x,
            q.x_hat + ||q|| epsilon_x].

The diagnostic replays actual canonical DiskANN expansion traces and reports:
  * point certificate: before an SSD read, can x itself be proven unable to
    improve the true top-k threshold?
  * direct-neighborhood certificate: after reading x's adjacency list, can every
    newly exposed neighbor be proven unable to improve that threshold?

The second quantity measures direct-result value of an expansion, NOT yet a
proof that the SSD read can be skipped safely for graph navigation: a bad
neighbor could still be a useful bridge. That distinction is intentionally
preserved in the output.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

LS = (40, 80, 160, 320)
SENTINEL = 0xFFFFFFFF


def read_bin_at(path: Path, dtype, offset=0):
    dt = np.dtype(dtype)
    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated matrix header")
        rows, cols = struct.unpack("<II", raw)
        arr = np.fromfile(f, dtype=dt, count=rows * cols)
    if arr.size != rows * cols:
        raise ValueError(f"{path}: truncated matrix payload")
    return arr.reshape(rows, cols)


def fbin_memmap(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: bad fbin size")
    return np.memmap(path, mode="r", dtype="<f4", offset=8, shape=(rows, dim))


def load_pq(pivots_path: Path, codes_path: Path):
    offsets_table = read_bin_at(pivots_path, "<u8", 0).reshape(-1)
    if offsets_table.size != 4:
        raise ValueError("PQ pivot offset table must have four entries")
    pivots = read_bin_at(pivots_path, "<f4", int(offsets_table[0])).astype(np.float32)
    centroid = read_bin_at(pivots_path, "<f4", int(offsets_table[1])).reshape(-1).astype(np.float32)
    chunk_offsets = read_bin_at(pivots_path, "<u4", int(offsets_table[2])).reshape(-1).astype(np.int64)
    if centroid.size != pivots.shape[1]:
        raise ValueError("PQ centroid dimension mismatch")
    if np.any(centroid != 0):
        pivots = pivots + centroid[None, :]
    if chunk_offsets[0] != 0 or chunk_offsets[-1] != pivots.shape[1]:
        raise ValueError("bad PQ chunk offsets")
    codes = read_bin_at(codes_path, "u1", 0)
    if codes.shape[1] != len(chunk_offsets) - 1:
        raise ValueError("PQ code/chunk mismatch")
    return pivots, chunk_offsets, codes


def graph_header(path: Path):
    with path.open("rb") as f:
        first = f.read(4096)
    vals = struct.unpack_from("<10Q", first, 8)
    block_size = struct.unpack_from("<Q", first, 88)[0] or 4096
    return {
        "n": int(vals[0]),
        "dim": int(vals[1]),
        "node_len": int(vals[3]),
        "nodes_per_block": int(vals[4]),
        "assoc": int(vals[9]),
        "block": int(block_size),
    }


class Graph:
    def __init__(self, path: Path):
        self.path = path
        self.h = graph_header(path)
        self.raw = np.memmap(path, mode="r", dtype="u1")
        self.vec_bytes = self.h["dim"] * 4
        self.cache = {}

    def neighbors(self, vid: int):
        cached = self.cache.get(vid)
        if cached is not None:
            return cached
        h = self.h
        if h["nodes_per_block"] > 0:
            sector = 1 + vid // h["nodes_per_block"]
            node_in = vid % h["nodes_per_block"]
            off = sector * h["block"] + node_in * h["node_len"]
        else:
            sectors = math.ceil(h["node_len"] / h["block"])
            off = (1 + vid * sectors) * h["block"]
        n_off = off + self.vec_bytes
        cnt = int(np.frombuffer(self.raw[n_off:n_off+4], dtype="<u4", count=1)[0])
        start = n_off + 4
        nbrs = np.frombuffer(self.raw[start:start + 4 * cnt], dtype="<u4", count=cnt).copy()
        self.cache[vid] = nbrs
        return nbrs


def compute_eps(base, pivots, chunk_offsets, codes, cache: Path | None):
    n, dim = base.shape
    if codes.shape[0] != n or pivots.shape[1] != dim:
        raise ValueError("base/PQ shape mismatch")
    if cache is not None and cache.exists():
        eps = np.load(cache)
        if eps.shape == (n,):
            print(f"loaded epsilon cache {cache}", flush=True)
            return eps.astype(np.float32, copy=False)

    eps = np.empty(n, dtype=np.float32)
    block = 8192
    nchunks = codes.shape[1]
    for s in range(0, n, block):
        e = min(n, s + block)
        xb = np.asarray(base[s:e], dtype=np.float32)
        cb = codes[s:e]
        ss = np.zeros(e - s, dtype=np.float64)
        for j in range(nchunks):
            a, b = int(chunk_offsets[j]), int(chunk_offsets[j+1])
            recon = pivots[cb[:, j], a:b]
            d = xb[:, a:b] - recon
            ss += np.einsum("ij,ij->i", d, d, optimize=True)
        eps[s:e] = np.sqrt(ss).astype(np.float32)
        if s == 0 or e % 100000 < block or e == n:
            print(f"pq_residuals={e}/{n}", flush=True)

    # Add one outward float32 ulp so later <= comparisons remain conservative.
    eps = np.nextafter(eps, np.float32(np.inf), dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, eps)
    return eps


def lut_for_query(q, pivots, chunk_offsets):
    nchunks = len(chunk_offsets) - 1
    lut = np.empty((nchunks, pivots.shape[0]), dtype=np.float32)
    for j in range(nchunks):
        a, b = int(chunk_offsets[j]), int(chunk_offsets[j+1])
        lut[j] = pivots[:, a:b] @ q[a:b]
    return lut


def pq_scores(ids, lut, codes):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty(0, dtype=np.float32)
    cb = codes[ids]
    out = np.zeros(ids.size, dtype=np.float32)
    rows = np.arange(ids.size)
    for j in range(codes.shape[1]):
        out += lut[j, cb[:, j]]
    return out


def gt_ids(path: Path):
    with path.open("rb") as f:
        rows, k = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows*k)
    if a.size != rows*k:
        raise ValueError("truncated GT")
    return a.reshape(rows, k)


def rate(a, b):
    return float(a / b) if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    for name in ("base", "queries", "gt5000", "pq-pivots", "pq-codes", "disk-index", "trace-dir", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--epsilon-cache", type=Path)
    ap.add_argument("--query-offset", type=int, default=9000)
    ap.add_argument("--gt-offset", type=int, default=4000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    base = fbin_memmap(args.base.resolve())
    queries_all = fbin_memmap(args.queries.resolve())
    queries = np.asarray(
        queries_all[args.query_offset:args.query_offset + 1000],
        dtype=np.float32,
    )
    gt_all = gt_ids(args.gt5000.resolve())
    gt = gt_all[args.gt_offset:args.gt_offset + len(queries)]
    if gt.shape[1] < args.k:
        raise ValueError("GT does not contain requested k")

    pivots, chunk_offsets, codes = load_pq(args.pq_pivots.resolve(), args.pq_codes.resolve())
    eps = compute_eps(
        base, pivots, chunk_offsets, codes,
        args.epsilon_cache.resolve() if args.epsilon_cache else None,
    )
    graph = Graph(args.disk_index.resolve())
    if graph.h["n"] != base.shape[0] or graph.h["dim"] != base.shape[1]:
        raise ValueError("graph/base mismatch")

    # Conservative fp16 storage estimate: round every radius upward.
    eps16 = eps.astype(np.float16)
    low = eps16.astype(np.float32) < eps
    eps16[low] = np.nextafter(eps16[low], np.float16(np.inf), dtype=np.float16)
    eps16f = eps16.astype(np.float32)

    result = {
        "certificate": {
            "formula": "upper(q,x)=q dot xhat + ||q|| * epsilon_x",
            "epsilon": "L2 distance from original vector to exact DiskANN PQ reconstruction",
            "guarantee": "upper(q,x) >= exact inner product q dot x",
        },
        "epsilon": {
            "float32_bytes_for_1M": int(eps.nbytes),
            "float16_upward_bytes_for_1M": int(eps16.nbytes),
            "min": float(eps.min()),
            "median": float(np.median(eps)),
            "mean": float(eps.mean()),
            "p95": float(np.quantile(eps, .95)),
            "max": float(eps.max()),
            "fp16_upward_mean_inflation": float(np.mean(eps16f - eps)),
            "fp16_upward_p95_inflation": float(np.quantile(eps16f - eps, .95)),
        },
        "query_norm": {
            "min": float(np.linalg.norm(queries, axis=1).min()),
            "median": float(np.median(np.linalg.norm(queries, axis=1))),
            "mean": float(np.linalg.norm(queries, axis=1).mean()),
            "max": float(np.linalg.norm(queries, axis=1).max()),
        },
        "by_L": {},
    }

    tol = 2e-5
    max_bound_violation = 0.0

    for L in LS:
        trace_path = args.trace_dir.resolve() / f"heldout.L{L}.jsonl"
        recs = [json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()]
        if len(recs) != len(queries):
            raise ValueError(f"L{L}: trace/query count mismatch")

        counts = {
            "expansions": 0,
            "point_final_reject": 0,
            "point_current_reject": 0,
            "point_current_eligible": 0,
            "neighbor_sets": 0,
            "neighbors_total": 0,
            "neighbors_final_rejected": 0,
            "neighbors_current_rejected": 0,
            "neighbor_all_final_reject": 0,
            "neighbor_all_current_reject": 0,
            "neighbor_current_eligible": 0,
            "true_topk_neighbor_exposures": 0,
            "certified_all_with_true_topk_neighbor": 0,
        }
        stage = [
            {"expansions":0, "point_final_reject":0, "neighbor_all_final_reject":0}
            for _ in range(4)
        ]

        for qi, rec in enumerate(recs):
            q = queries[qi]
            qnorm = float(np.linalg.norm(q))
            lut = lut_for_query(q, pivots, chunk_offsets)

            truth = gt[qi, :args.k].astype(np.int64)
            truth_vec = np.asarray(base[truth], dtype=np.float32)
            truth_scores = truth_vec @ q
            tau_final = float(np.min(truth_scores))
            truth_set = set(int(x) for x in truth)

            ids = [int(x) for x in rec["ids"]]
            exact_seen = []
            for pos, x in enumerate(ids):
                frac = min(3, (4 * pos) // max(1, len(ids)))
                counts["expansions"] += 1
                stage[frac]["expansions"] += 1

                sx = float(pq_scores([x], lut, codes)[0])
                ux = sx + qnorm * float(eps[x])
                exact_x = float(np.dot(np.asarray(base[x], dtype=np.float32), q))
                max_bound_violation = max(max_bound_violation, exact_x - ux)

                if ux < tau_final - tol:
                    counts["point_final_reject"] += 1
                    stage[frac]["point_final_reject"] += 1

                if len(exact_seen) >= args.k:
                    tau_current = float(np.partition(np.asarray(exact_seen), -args.k)[-args.k])
                    counts["point_current_eligible"] += 1
                    if ux < tau_current - tol:
                        counts["point_current_reject"] += 1
                else:
                    tau_current = None

                nbrs = graph.neighbors(x).astype(np.int64, copy=False)
                counts["neighbor_sets"] += 1
                counts["neighbors_total"] += int(nbrs.size)
                if nbrs.size:
                    sn = pq_scores(nbrs, lut, codes)
                    un = sn.astype(np.float64) + qnorm * eps[nbrs].astype(np.float64)
                    rejected_final = un < tau_final - tol
                    counts["neighbors_final_rejected"] += int(rejected_final.sum())
                    all_final = bool(np.all(rejected_final))
                    if all_final:
                        counts["neighbor_all_final_reject"] += 1
                        stage[frac]["neighbor_all_final_reject"] += 1

                    has_truth = any(int(y) in truth_set for y in nbrs)
                    if has_truth:
                        counts["true_topk_neighbor_exposures"] += 1
                        if all_final:
                            counts["certified_all_with_true_topk_neighbor"] += 1

                    if tau_current is not None:
                        counts["neighbor_current_eligible"] += 1
                        rejected_current = un < tau_current - tol
                        counts["neighbors_current_rejected"] += int(rejected_current.sum())
                        if bool(np.all(rejected_current)):
                            counts["neighbor_all_current_reject"] += 1

                exact_seen.append(exact_x)

        result["by_L"][str(L)] = {
            "counts": counts,
            "rates": {
                "point_final_reject_fraction": rate(counts["point_final_reject"], counts["expansions"]),
                "point_current_reject_fraction_after_k_validated": rate(counts["point_current_reject"], counts["point_current_eligible"]),
                "individual_neighbor_final_reject_fraction": rate(counts["neighbors_final_rejected"], counts["neighbors_total"]),
                "neighbor_set_all_final_reject_fraction": rate(counts["neighbor_all_final_reject"], counts["neighbor_sets"]),
                "neighbor_set_all_current_reject_fraction_after_k_validated": rate(counts["neighbor_all_current_reject"], counts["neighbor_current_eligible"]),
            },
            "stage_quartiles": [
                {
                    "quartile": i + 1,
                    "expansions": d["expansions"],
                    "point_final_reject_fraction": rate(d["point_final_reject"], d["expansions"]),
                    "neighbor_set_all_final_reject_fraction": rate(d["neighbor_all_final_reject"], d["expansions"]),
                }
                for i, d in enumerate(stage)
            ],
        }

    result["safety_checks"] = {
        "max_exact_minus_upper_bound": float(max_bound_violation),
        "neighbor_all_certificate_true_topk_violations": int(sum(
            result["by_L"][str(L)]["counts"]["certified_all_with_true_topk_neighbor"] for L in LS
        )),
        "interpretation": "negative/zero exact-minus-upper and zero top-k-neighbor violations are expected",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
