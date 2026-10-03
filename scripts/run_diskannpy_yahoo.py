#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import diskannpy
import numpy as np

COMPLEXITIES = (20, 40, 60, 100, 200)
K = 10


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int = K) -> float:
    ids = np.asarray(ids[:, :k], dtype=np.uint32)
    gt = np.asarray(gt[:, :k], dtype=np.uint32)
    hits = 0
    for a, b in zip(ids, gt):
        hits += len(set(map(int, a)) & set(map(int, b)))
    return 100.0 * hits / (len(ids) * k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.index_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    build_start = time.perf_counter()
    diskannpy.build_memory_index(
        data=str(args.base),
        vector_dtype=np.float32,
        distance_metric="l2",
        index_directory=str(args.index_dir),
        complexity=100,
        graph_degree=64,
        num_threads=4,
        alpha=1.2,
        use_pq_build=False,
        num_pq_bytes=0,
        use_opq=False,
        index_prefix="ann",
    )
    build_s = time.perf_counter() - build_start

    graph = args.index_dir / "ann"
    payload = args.index_dir / "ann.data"
    if not graph.is_file() or not payload.is_file():
        raise FileNotFoundError("diskannpy did not produce ann and ann.data")

    q = np.asarray(np.load(args.queries), dtype=np.float32, order="C")
    gt = np.asarray(np.load(args.gt), dtype=np.uint32, order="C")
    index = diskannpy.StaticMemoryIndex(
        index_directory=str(args.index_dir),
        num_threads=1,
        initial_search_complexity=max(COMPLEXITIES),
        index_prefix="ann",
    )

    result = {
        "diskannpy_version": getattr(diskannpy, "__version__", "unknown"),
        "build_seconds": build_s,
        "graph_bytes": graph.stat().st_size,
        "payload_bytes": payload.stat().st_size,
        "build": {
            "distance_metric": "l2",
            "complexity": 100,
            "graph_degree": 64,
            "num_threads": 4,
            "alpha": 1.2,
            "use_pq_build": False,
            "num_pq_bytes": 0,
        },
        "search": {},
    }

    # Warm once without including it in timing.
    index.batch_search(q[:8], k_neighbors=K, complexity=60, num_threads=1)

    for complexity in COMPLEXITIES:
        reps = []
        ids_for_recall = None
        for rep in range(3):
            t0 = time.perf_counter()
            response = index.batch_search(
                q, k_neighbors=K, complexity=complexity, num_threads=1
            )
            elapsed = time.perf_counter() - t0
            ids = np.asarray(response.identifiers, dtype=np.uint32)
            if ids_for_recall is None:
                ids_for_recall = ids.copy()
            else:
                if not np.array_equal(ids_for_recall, ids):
                    raise AssertionError("DiskANN results changed between timing repetitions")
            reps.append({
                "elapsed_seconds": elapsed,
                "qps": len(q) / elapsed,
            })

        np.save(args.out / f"diskannpy-c{complexity}-ids.npy", ids_for_recall)
        result["search"][str(complexity)] = {
            "recall_at_10_percent": recall_at_k(ids_for_recall, gt),
            "repetitions": reps,
            "qps_mean": float(np.mean([x["qps"] for x in reps])),
            "qps_std": float(np.std([x["qps"] for x in reps])),
        }

    (args.out / "diskannpy-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
