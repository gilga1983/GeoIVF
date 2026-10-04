#!/usr/bin/env python3
"""Reconstruct the identity stream of the MedRAG-Zipf workload.

The original 10k paraphrase text is not publicly released. This script pins
everything that *is* specified by the Proximity/CatapultDB papers:
- MIRAGE PubMedQA* test set: 500 questions;
- 10,000 iid draws;
- discrete Zipf rank weights proportional to rank^-0.8;
- no temporal-locality construction.

The reconstruction seed is deliberately recorded. Seed 1367 was selected
because it produces all 500 source questions at least once and a hottest
question count of 699, matching the paper's descriptive "about 700" statistic.
It is not claimed to be the authors' unpublished random seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_BASE = 500
N_QUERIES = 10_000
ZIPF_EXPONENT = 0.8
RECONSTRUCTION_SEED = 1367
MIRAGE_COMMIT = "392943af99cd94cafd50a0de2e7fca24bbf65494"
MIRAGE_PATH = "rawdata/pubmedqa/data/test_set.json"


def load_questions(path: Path):
    obj = json.loads(path.read_text())
    rows = []
    if isinstance(obj, dict):
        items = list(obj.items())
        for key, value in items:
            if not isinstance(value, dict):
                raise ValueError("unexpected PubMedQA record")
            q = value.get("QUESTION") or value.get("question")
            if not isinstance(q, str) or not q.strip():
                raise ValueError(f"missing question for {key}")
            rows.append((str(key), q.strip()))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if not isinstance(value, dict):
                raise ValueError("unexpected PubMedQA record")
            key = value.get("id", i)
            q = value.get("QUESTION") or value.get("question")
            if not isinstance(q, str) or not q.strip():
                raise ValueError(f"missing question for {key}")
            rows.append((str(key), q.strip()))
    else:
        raise ValueError("unexpected PubMedQA JSON root")

    if len(rows) != N_BASE:
        raise ValueError(f"expected {N_BASE} MIRAGE PubMedQA* questions, got {len(rows)}")
    if len({q for _, q in rows}) != N_BASE:
        raise ValueError("base questions are not unique")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmedqa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=RECONSTRUCTION_SEED)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base = load_questions(args.pubmedqa)

    ranks = np.arange(1, N_BASE + 1, dtype=np.float64)
    weights = ranks ** (-ZIPF_EXPONENT)
    probs = weights / weights.sum()

    rng = np.random.default_rng(args.seed)
    identities = rng.choice(N_BASE, size=N_QUERIES, replace=True, p=probs).astype(np.uint16)
    counts = np.bincount(identities, minlength=N_BASE)
    if np.any(counts == 0):
        raise ValueError(
            f"reconstruction seed {args.seed} omits {int(np.sum(counts == 0))} source questions"
        )

    np.save(args.out / "source_indices.npy", identities)

    occurrence = np.zeros(N_BASE, dtype=np.int64)
    with (args.out / "stream.jsonl").open("w") as f:
        for stream_pos, source_idx in enumerate(identities):
            source_idx = int(source_idx)
            occ = int(occurrence[source_idx])
            occurrence[source_idx] += 1
            source_id, question = base[source_idx]
            f.write(json.dumps({
                "stream_pos": stream_pos,
                "source_index": source_idx,
                "source_id": source_id,
                "source_rank": source_idx + 1,
                "occurrence": occ,
                "base_question": question,
            }, ensure_ascii=False) + "\n")

    with (args.out / "base_questions.jsonl").open("w") as f:
        for i, (source_id, question) in enumerate(base):
            f.write(json.dumps({
                "source_index": i,
                "source_id": source_id,
                "source_rank": i + 1,
                "question": question,
                "stream_count": int(counts[i]),
                "sampling_probability": float(probs[i]),
            }, ensure_ascii=False) + "\n")

    source_sha = hashlib.sha256(args.pubmedqa.read_bytes()).hexdigest()
    stats = {
        "name": "MedRAG-Zipf reconstruction identity stream",
        "paper_recipe": {
            "source_questions": N_BASE,
            "queries": N_QUERIES,
            "zipf_exponent": ZIPF_EXPONENT,
            "sampling": "iid discrete rank-weighted draws; no temporal locality",
        },
        "source": {
            "repository": "gzxiong/MIRAGE",
            "commit": MIRAGE_COMMIT,
            "path": MIRAGE_PATH,
            "sha256": source_sha,
        },
        "reconstruction": {
            "numpy_seed": args.seed,
            "seed_status": "reconstruction choice; authors' workload seed is unpublished",
            "rank_assignment": "MIRAGE JSON insertion order",
        },
        "observed": {
            "unique_source_questions": int(np.count_nonzero(counts)),
            "hottest_count": int(counts.max()),
            "coldest_count": int(counts.min()),
            "median_count": float(np.median(counts)),
            "top10_queries_fraction": float(np.sort(counts)[-10:].sum() / N_QUERIES),
            "top100_queries_fraction": float(np.sort(counts)[-100:].sum() / N_QUERIES),
        },
        "paraphrase_requirement": {
            "one_unique_paraphrase_per_stream_occurrence": True,
            "paper_model": "gpt-4o-2024-08-06",
            "paper_validation": "global text uniqueness and same no-cache RAG answer across rephrasings",
            "status": "not generated by this script",
        },
    }
    (args.out / "identity-manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
