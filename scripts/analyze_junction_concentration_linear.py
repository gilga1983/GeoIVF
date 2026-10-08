#!/usr/bin/env python3
"""Rank junctions on earlier DiskANN queries; score them on later queries.

For the upper 5% of the corpus, small training workloads may never visit
enough distinct vertices to fill the requested budget. Unobserved IDs tie at
zero training visits, and we break this tie reproducibly using random sampling.
The output marks the observed-support boundary for transparent visualization.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import random
from pathlib import Path

BASE_FRACTIONS = (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
                  0.016, 0.02, 0.03, 0.04, 0.05)
SEED = 20261008


def analyze(path: Path, corpus_size: int, skip: int,
            train_rows: int, total_rows: int, search_l: int):
    train, test = Counter(), Counter()
    train_visits = test_visits = 0
    raw_expansions = 0
    nrows = 0
    with path.open() as stream:
        for qid, line in enumerate(stream):
            if qid >= total_rows:
                raise ValueError("too many traces")
            rec = json.loads(line)
            if int(rec["query"]) != qid:
                raise ValueError("trace rows are out of order")
            ids = [int(v) for v in rec["ids"]]
            if any(v < 0 or v >= corpus_size for v in ids):
                raise ValueError(f"out-of-range vertex ID in query {qid}")
            raw_expansions += len(ids)
            suffix = ids[skip:]
            if qid < train_rows:
                train.update(suffix)
                train_visits += len(suffix)
            else:
                test.update(suffix)
                test_visits += len(suffix)
            nrows += 1
    if nrows != total_rows or min(train_visits, test_visits) <= 0:
        raise ValueError("incomplete/empty trace stream")

    # Select vertices according to previous traffic ONLY.
    # All zero-frequency IDs tie; a reproducible uniform random tie-break
    # eliminates dependence on the corpus's ordering of vector IDs.
    ranked = sorted(train, key=lambda v: (-train[v], v))
    observed = len(ranked)
    required = min(corpus_size, round(max(BASE_FRACTIONS) * corpus_size))
    selected = ranked[:required]
    if len(selected) < required:
        chosen = set(ranked)
        rng = random.Random(SEED)
        while len(selected) < required:
            v = rng.randrange(corpus_size)
            if v in chosen:
                continue
            chosen.add(v)
            selected.append(v)
    scores = []
    visits = 0
    for v in selected:
        visits += test.get(v, 0)
        scores.append(visits / test_visits)

    support_fraction = observed / corpus_size
    fractions = sorted(set(BASE_FRACTIONS + (support_fraction,)))
    curve = []
    for fraction in fractions:
        k = round(fraction * corpus_size)
        if k > required:
            continue
        curve.append({
            "corpus_fraction": fraction,
            "vertices": k,
            "heldout_visit_share": (scores[k - 1] if k else 0.0),
            "ids_not_seen_in_training": max(0, k - observed),
        })
    if any(a["heldout_visit_share"] > b["heldout_visit_share"] + 1e-12
           for a, b in zip(curve, curve[1:])):
        raise ValueError("curve must be monotonic")
    return {
        "method": "ordinary DiskANN: first 2500 query traces rank junctions; next 2500 evaluate",
        "selection_for_zero_visit_ids": "reproducible random tie break",
        "zero_visit_tie_seed": SEED,
        "corpus_size": corpus_size,
        "train_queries": train_rows,
        "test_queries": total_rows - train_rows,
        "search_L": search_l,
        "beam_width": 8,
        "skip_first_expansions": skip,
        "total_train_suffix_visits": train_visits,
        "total_test_suffix_visits": test_visits,
        "train_distinct_vertices": observed,
        "train_observed_support_fraction": support_fraction,
        "test_distinct_vertices": len(test),
        "mean_expansions_per_query": raw_expansions / total_rows,
        "curve": curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--corpus-size", type=int, required=True)
    ap.add_argument("--skip-prefix", type=int, default=20)
    ap.add_argument("--train-rows", type=int, default=2500)
    ap.add_argument("--total-rows", type=int, default=5000)
    ap.add_argument("--search-l", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not 0 < a.train_rows < a.total_rows or a.corpus_size < 1 or a.skip_prefix < 0:
        ap.error("invalid analysis configuration")
    obj = analyze(a.trace, a.corpus_size, a.skip_prefix,
                  a.train_rows, a.total_rows, a.search_l)
    obj["dataset"] = a.dataset
    obj["trace_file"] = str(a.trace)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.dataset}.json").write_text(json.dumps(obj, indent=2) + "\n")
    fields = ("corpus_fraction", "vertices", "heldout_visit_share",
              "ids_not_seen_in_training")
    with (a.out / f"{a.dataset}.csv").open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=fields)
        w.writeheader()
        w.writerows(obj["curve"])
    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
