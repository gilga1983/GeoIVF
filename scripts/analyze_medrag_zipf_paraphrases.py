#!/usr/bin/env python3
"""Quality diagnostics for reconstructed MedRAG-Zipf paraphrases."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def embed(texts, tokenizer, model, device, batch_size=64):
    out = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s+batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            y = model(**enc).last_hidden_state[:, 0, :].float()
        out.append(y.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def normalize(a):
    n = np.linalg.norm(a.astype(np.float64), axis=1)
    if np.any(n <= 0) or np.any(~np.isfinite(n)):
        raise ValueError("bad embedding norm")
    return np.asarray(a / n[:, None], dtype=np.float32)


def percentiles(a):
    return {
        "min": float(np.min(a)),
        "p01": float(np.quantile(a, .01)),
        "p05": float(np.quantile(a, .05)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.quantile(a, .95)),
        "max": float(np.max(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrases", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="ncbi/MedCPT-Query-Encoder")
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.paraphrases.read_text().splitlines() if x.strip()]
    base_rows = [json.loads(x) for x in args.base.read_text().splitlines() if x.strip()]
    if len(base_rows) != 500:
        raise ValueError(f"expected 500 base questions, got {len(base_rows)}")
    if not rows:
        raise ValueError("no paraphrases")

    base_texts = [x["question"] for x in base_rows]
    para_texts = [x["paraphrase"] for x in rows]
    source = np.asarray([int(x["source_index"]) for x in rows], dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to("cuda")
    model.eval()

    base_emb = normalize(embed(base_texts, tokenizer, model, "cuda"))
    para_emb = normalize(embed(para_texts, tokenizer, model, "cuda"))
    sims = para_emb @ base_emb.T
    own = sims[np.arange(len(rows)), source]

    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    source_rank = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        source_rank[i] = int(np.where(order[i] == source[i])[0][0]) + 1

    counts = Counter(source.tolist())
    group_spread = []
    groups = defaultdict(list)
    for i, sid in enumerate(source):
        groups[int(sid)].append(i)
    for sid, inds in groups.items():
        if len(inds) < 2:
            continue
        e = para_emb[inds]
        center = normalize(e.mean(axis=0, keepdims=True))[0]
        group_spread.extend((e @ center).tolist())

    lengths_base = np.asarray([len(base_texts[s].split()) for s in source], dtype=np.float64)
    lengths_para = np.asarray([len(x.split()) for x in para_texts], dtype=np.float64)
    exact_dupes = len(para_texts) - len({x.casefold() for x in para_texts})

    result = {
        "rows": len(rows),
        "sources_represented": len(counts),
        "text_quality": {
            "casefolded_duplicate_count": exact_dupes,
            "question_mark_fraction": float(np.mean([x.endswith("?") for x in para_texts])),
            "unchanged_from_source_count": int(sum(
                p.casefold() == base_texts[s].casefold() for p, s in zip(para_texts, source)
            )),
            "word_length_ratio": percentiles(lengths_para / np.maximum(lengths_base, 1)),
        },
        "medcpt_geometry": {
            "source_cosine": percentiles(own),
            "source_is_nearest_base_fraction": float(np.mean(top1 == source)),
            "source_in_top5_base_fraction": float(np.mean(source_rank <= 5)),
            "source_in_top10_base_fraction": float(np.mean(source_rank <= 10)),
            "source_rank": percentiles(source_rank.astype(np.float64)),
            "within_cluster_cosine_to_cluster_centroid": (
                percentiles(np.asarray(group_spread, dtype=np.float64))
                if group_spread else None
            ),
        },
        "model": args.model,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
