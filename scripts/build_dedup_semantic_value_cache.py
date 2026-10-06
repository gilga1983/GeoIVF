#!/usr/bin/env python3
"""Build deduplicated value-ID semantic caches from actual prior DiskANN results.

Everything except cache population is held fixed.

Policies:
  skipdup:
    Consider only the completed search's top-1 result. If it is already in the
    cache, do nothing. Otherwise insert it.

  nextdistinct:
    Scan the completed top-10 result list in rank order and insert the first ID
    not already in the cache. If all ten are already present, do nothing.

The cache is insertion-FIFO for this isolated experiment. Lookup does not alter
replacement state. Each cached anchor retains the sibling result set from the
request that inserted it. For an evaluated query, scan unique cached anchor IDs
with DiskANN's existing resident PQ codes, choose the best anchor, and emit that
anchor followed by its page-resident siblings.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np

from analyze_pq_semantic_filter import fbin_memmap, load_pq, lut_for_query

WARM0 = 5000
EVAL0 = 9000
EVAL_N = 1000


def read_ids(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rows, k = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows * k)
    if a.size != rows * k:
        raise ValueError("truncated result dump")
    return a.reshape(rows, k)


def unique_results(row: np.ndarray) -> list[int]:
    seen = set()
    out = []
    for raw in row:
        x = int(raw)
        if x not in seen:
            seen.add(x)
            out.append(x)
    if len(out) < 10:
        raise ValueError("producer row contains duplicate/insufficient IDs")
    return out[:10]


def write_rows(path: Path, a: np.ndarray) -> None:
    a = np.asarray(a, dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        a.tofile(f)


class UniqueCache:
    def __init__(self, capacity: int, policy: str):
        self.capacity = capacity
        self.policy = policy
        self.order: deque[int] = deque()
        self.entries: dict[int, list[int]] = {}
        self.inserts = 0
        self.skips = 0
        self.evictions = 0
        self.chosen_ranks: list[int] = []

    def insert_result_set(self, result_ids: list[int]) -> None:
        chosen = None
        rank = None
        if self.policy == "skipdup":
            cand = result_ids[0]
            if cand in self.entries:
                self.skips += 1
                return
            chosen = cand
            rank = 1
        elif self.policy == "nextdistinct":
            for i, cand in enumerate(result_ids, start=1):
                if cand not in self.entries:
                    chosen = cand
                    rank = i
                    break
            if chosen is None:
                self.skips += 1
                return
        else:
            raise ValueError(self.policy)

        # Put chosen anchor first, then the remaining successful results.
        siblings = [chosen] + [x for x in result_ids if x != chosen]
        if len(siblings) < 10:
            raise RuntimeError("insufficient page siblings")
        siblings = siblings[:10]

        if len(self.entries) >= self.capacity:
            victim = self.order.popleft()
            del self.entries[victim]
            self.evictions += 1

        self.order.append(chosen)
        self.entries[chosen] = siblings
        self.inserts += 1
        self.chosen_ranks.append(rank)

    def ids(self) -> np.ndarray:
        # Preserve insertion order only for deterministic tie breaking.
        return np.fromiter(self.order, dtype=np.int64, count=len(self.order))


def score_values(q: np.ndarray, ids: np.ndarray, pivots, offsets, db_codes) -> np.ndarray:
    lut = lut_for_query(q, pivots, offsets)
    cb = db_codes[ids]
    out = np.zeros(len(ids), dtype=np.float32)
    for j in range(db_codes.shape[1]):
        out += lut[j, cb[:, j]]
    return out


def build_policy(
    policy: str,
    cap: int,
    queries: np.ndarray,
    results: np.ndarray,
    pivots,
    offsets,
    db_codes,
):
    cache = UniqueCache(cap, policy)
    output = np.empty((EVAL_N, 10), dtype=np.uint32)
    scan_seconds = 0.0
    occupancy = []
    selected = np.empty(EVAL_N, dtype=np.uint32)

    # Replay q=5000..9999. Query i consults only completed earlier requests.
    for rel in range(results.shape[0]):
        abs_q = WARM0 + rel

        if abs_q >= EVAL0:
            ei = abs_q - EVAL0
            ids = cache.ids()
            if len(ids) == 0:
                raise RuntimeError("empty cache at evaluation")
            t0 = time.perf_counter()
            scores = score_values(
                np.asarray(queries[abs_q], dtype=np.float32),
                ids,
                pivots,
                offsets,
                db_codes,
            )
            scan_seconds += time.perf_counter() - t0
            anchor = int(ids[int(np.argmax(scores))])
            selected[ei] = anchor
            output[ei] = np.asarray(cache.entries[anchor], dtype=np.uint32)
            occupancy.append(len(ids))

        cache.insert_result_set(unique_results(results[rel]))

    rank_hist = {}
    for r in cache.chosen_ranks:
        rank_hist[str(r)] = rank_hist.get(str(r), 0) + 1

    return output, selected, {
        "capacity": cap,
        "directory_bytes": cap * 4,
        "python_scan_us_per_eval_query": 1e6 * scan_seconds / EVAL_N,
        "mean_eval_occupancy": float(np.mean(occupancy)),
        "min_eval_occupancy": int(np.min(occupancy)),
        "max_eval_occupancy": int(np.max(occupancy)),
        "inserts": cache.inserts,
        "skipped_insertions": cache.skips,
        "evictions": cache.evictions,
        "insert_rank_histogram": rank_hist,
        "mean_insert_rank": float(np.mean(cache.chosen_ranks)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for n in ("queries", "pq-pivots", "pq-codes", "results", "out-dir"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--capacities", default="512,2048")
    args = ap.parse_args()

    caps = sorted({int(x) for x in args.capacities.split(",") if x.strip()})
    queries = fbin_memmap(args.queries.resolve())
    pivots, offsets, db_codes = load_pq(
        args.pq_pivots.resolve(), args.pq_codes.resolve()
    )
    results = read_ids(args.results.resolve())

    if queries.shape != (10000, 768):
        raise ValueError(f"unexpected query shape {queries.shape}")
    if results.shape != (5000, 10):
        raise ValueError(f"unexpected result shape {results.shape}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy": "unique value-ID cache; isolated population policy comparison",
        "replacement": "insertion FIFO; lookup does not touch state",
        "producer": str(args.results),
        "capacities": {},
    }

    for cap in caps:
        manifest["capacities"][str(cap)] = {}
        selected_by_policy = {}

        for policy in ("skipdup", "nextdistinct"):
            rows, selected, meta = build_policy(
                policy, cap, queries, results, pivots, offsets, db_codes
            )
            p = args.out_dir / f"{policy}-c{cap}.bin"
            write_rows(p, rows)
            meta["file"] = p.name
            manifest["capacities"][str(cap)][policy] = meta
            selected_by_policy[policy] = selected

            print(json.dumps({
                "capacity": cap,
                "policy": policy,
                "directory_kib": meta["directory_bytes"] / 1024,
                "mean_occupancy": meta["mean_eval_occupancy"],
                "skips": meta["skipped_insertions"],
                "mean_insert_rank": meta["mean_insert_rank"],
                "python_scan_us": meta["python_scan_us_per_eval_query"],
                "rank_hist": meta["insert_rank_histogram"],
            }), flush=True)

        manifest["capacities"][str(cap)]["selected_anchor_agreement_fraction"] = float(
            np.mean(
                selected_by_policy["skipdup"]
                == selected_by_policy["nextdistinct"]
            )
        )

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
