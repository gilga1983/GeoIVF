#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ALPHAS = (0.0, 0.75, 1.0, 1.25)


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    q = np.asarray(np.load(args.queries), dtype=np.float32, order="C")
    gt = np.asarray(np.load(args.gt), dtype=np.uint32, order="C")
    if len(q) != len(gt):
        raise ValueError("query/ground-truth population mismatch")
    if len(q) != 1000:
        raise ValueError(f"expected 1000 canonical Yahoo queries, got {len(q)}")
    if args.length <= 0:
        raise ValueError("length must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    popularity_order = rng.permutation(len(q))
    uniforms = rng.random(args.length)

    np.save(args.out / "popularity_order.npy", popularity_order.astype(np.uint32))

    all_meta = {
        "seed": args.seed,
        "canonical_queries": int(len(q)),
        "replay_length": int(args.length),
        "alphas": list(ALPHAS),
        "streams": {},
    }

    ranks = np.arange(1, len(q) + 1, dtype=np.float64)
    for alpha in ALPHAS:
        weights = ranks ** (-alpha)
        probs = weights / weights.sum()
        cdf = np.cumsum(probs)
        rank_idx = np.searchsorted(cdf, uniforms, side="right")
        rank_idx = np.minimum(rank_idx, len(q) - 1)
        ids = popularity_order[rank_idx]

        replay_q = np.asarray(q[ids], dtype=np.float32, order="C")
        replay_gt = np.asarray(gt[ids], dtype=np.uint32, order="C")
        tag = alpha_tag(alpha)
        np.save(args.out / f"queries-z{tag}.npy", replay_q)
        np.save(args.out / f"gt-z{tag}.npy", replay_gt)
        np.save(args.out / f"source-ids-z{tag}.npy", ids.astype(np.uint32))

        counts = np.bincount(ids, minlength=len(q))
        sorted_counts = np.sort(counts)[::-1]
        meta = {
            "alpha": alpha,
            "unique_queries_observed": int(np.count_nonzero(counts)),
            "top1_fraction": float(sorted_counts[:1].sum() / args.length),
            "top10_fraction": float(sorted_counts[:10].sum() / args.length),
            "top100_fraction": float(sorted_counts[:100].sum() / args.length),
            "max_repetitions": int(sorted_counts[0]),
            "median_repetitions_nonzero": float(np.median(counts[counts > 0])),
        }
        all_meta["streams"][tag] = meta
        (args.out / f"meta-z{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")

    (args.out / "zipf-replay.json").write_text(json.dumps(all_meta, indent=2) + "\n")
    print(json.dumps(all_meta, indent=2))


if __name__ == "__main__":
    main()
