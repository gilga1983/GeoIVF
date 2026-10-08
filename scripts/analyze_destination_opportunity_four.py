#!/usr/bin/env python3
"""Causal exact-neighbor destination reuse opportunity across ANN workloads.

A bounded, unique FIFO stores the *previous query's exact rank-1 ID*, not a query,
result list, or per-query lookup key. Each measured query asks whether the
current exact top-10 includes one of these stored IDs, before the update.

Ground truth is an opportunity oracle only; deployed NavHints uses approximate
winners and never skips the usual DiskANN search.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np

CAPACITIES = (0, 16, 32, 64, 128, 256, 512, 1024, 2048)


def load_ids(path: Path):
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("truncated ground-truth header")
    rows, k = np.frombuffer(header, dtype="<u4")
    rows, k = int(rows), int(k)
    if rows <= 0 or k < 10:
        raise ValueError(f"expected >=10 neighbors: got ({rows}, {k})")
    length = path.stat().st_size
    only_ids = 8 + rows * k * 4
    if length not in (only_ids, only_ids + rows * k * 4):
        raise ValueError(f"ground-truth file has unexpected size {length} ({rows=}, {k=})")
    return np.memmap(path, dtype="<u4", mode="r", shape=(rows, k), offset=8)


def evaluate(gt, history_start: int, warmup: int, measured: int):
    """Independent per-capacity causal histories. No training data is replayed."""
    if history_start < 0 or min(warmup, measured) <= 0 or history_start + warmup + measured > len(gt):
        raise ValueError("invalid causal split")
    result = []
    for capacity in CAPACITIES:
        members = set()
        fifo = deque()
        top1 = top10 = 0
        attempted = measured
        novel = duplicate = 0
        for qi in range(history_start, history_start + warmup + measured):
            dest = int(gt[qi, 0])
            if qi >= history_start + warmup:
                top1 += dest in members
                top10 += any(int(x) in members for x in gt[qi, :10])
            if capacity:
                if dest not in members:
                    novel += 1
                    if len(fifo) >= capacity:
                        members.remove(fifo.popleft())
                    members.add(dest)
                    fifo.append(dest)
                else:
                    duplicate += 1
        assert len(fifo) == len(members) and len(fifo) <= capacity
        result.append({
            "capacity": capacity,
            "exact_top1_present_pct": 100.0 * top1 / attempted,
            "any_exact_top10_present_pct": 100.0 * top10 / attempted,
            "top1_present_requests": top1,
            "top10_present_requests": top10,
            "history_new_destinations": novel,
            "history_duplicate_destinations": duplicate,
        })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--history-start", type=int, default=0)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--measured", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    gt = load_ids(a.gt)
    rows = evaluate(gt, a.history_start, a.warmup, a.measured)
    obj = {
        "dataset": a.dataset,
        "source_gt": str(a.gt),
        "total_groundtruth_queries": len(gt),
        "history_start": a.history_start,
        "warmup_queries": a.warmup,
        "measured_queries": a.measured,
        "oracle": "previous exact rank-1 IDs (not deployed approximate results)",
        "policy": "bounded unique FIFO with skip duplicates; causal and query-key-free",
        "measure": "current exact top-1/top-10 contains at least one past rank-1 ID",
        "results": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.dataset}.json").write_text(json.dumps(obj, indent=2) + "\n")
    with (a.out / f"{a.dataset}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in obj.items() if k != "source_gt"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
