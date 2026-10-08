#!/usr/bin/env python3
"""Independent-cohort concentration curves for ordinary DiskANN traversals.

Vertices are ranked by post-prefix visits in the first half of a query stream
and scored only on later queries. This prevents hindsight/sampling artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

FRACTIONS = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.016, 0.02, 0.05)


def analyze(path: Path, corpus_size: int, skip: int, train_rows: int, total_rows: int):
    train, test = Counter(), Counter()
    train_visits = test_visits = 0
    raw_expansions = 0
    with path.open() as f:
        row_count = 0
        for row_id, line in enumerate(f):
            if row_id >= total_rows:
                raise ValueError("too many trace rows")
            row = json.loads(line)
            if int(row["query"]) != row_id:
                raise ValueError(f"unexpected trace query {row['query']} at row {row_id}")
            ids = list(map(int, row["ids"]))
            if any(v < 0 or v >= corpus_size for v in ids):
                raise ValueError(f"vertex outside corpus in row {row_id}")
            raw_expansions += len(ids)
            suffix = ids[skip:]
            if row_id < train_rows:
                train.update(suffix)
                train_visits += len(suffix)
            else:
                test.update(suffix)
                test_visits += len(suffix)
            row_count += 1
    if row_count != total_rows:
        raise ValueError(f"expected {total_rows} traces, found {row_count}")
    if min(train_visits, test_visits) <= 0:
        raise ValueError("empty post-prefix traversal traffic")
    ranking = sorted(train, key=lambda v: (-train[v], v))
    eval_ranking = sorted(test, key=lambda v: (-test[v], v))
    curve = []
    for f in FRACTIONS:
        k = max(1, round(f * corpus_size))
        # Do not guess how to rank never-visited (zero-count) vertices.
        heldout = None if k > len(ranking) else sum(test[v] for v in ranking[:k]) / test_visits
        in_sample = None if k > len(eval_ranking) else sum(test[v] for v in eval_ranking[:k]) / test_visits
        curve.append({"corpus_fraction": f, "vertices": k,
                      "heldout_visit_share": heldout,
                      "within_holdout_visit_share": in_sample})
    return {
        "method": "vanilla medoid-start DiskANN; ranks from first 2500 queries, held-out traffic from later 2500",
        "corpus_size": corpus_size,
        "train_queries": train_rows,
        "test_queries": total_rows - train_rows,
        "skip_first_expansions": skip,
        "search_L": 64,
        "beam_width": 8,
        "total_train_suffix_visits": train_visits,
        "total_test_suffix_visits": test_visits,
        "train_distinct_vertices": len(ranking),
        "test_distinct_vertices": len(eval_ranking),
        "mean_expansions_per_query": raw_expansions / total_rows,
        "curve": curve,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--corpus-size", type=int, required=True)
    p.add_argument("--skip-prefix", type=int, default=20)
    p.add_argument("--train-rows", type=int, default=2500)
    p.add_argument("--total-rows", type=int, default=5000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not (0 < a.train_rows < a.total_rows) or a.corpus_size < 1 or a.skip_prefix < 0:
        p.error("invalid analysis configuration")
    result = analyze(a.trace, a.corpus_size, a.skip_prefix, a.train_rows, a.total_rows)
    result.update(dataset=a.dataset, trace_file=str(a.trace))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.dataset}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.out / f"{a.dataset}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("corpus_fraction", "vertices", "heldout_visit_share",
                                          "within_holdout_visit_share"))
        w.writeheader()
        w.writerows(result["curve"])
    print(json.dumps({k: v for k, v in result.items() if k != "trace_file"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
