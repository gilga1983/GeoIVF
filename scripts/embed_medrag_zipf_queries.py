#!/usr/bin/env python3
"""Embed reconstructed MedRAG-Zipf queries with the official MedCPT query encoder."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DIM = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="ncbi/MedCPT-Query-Encoder")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.paraphrases.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError("no paraphrases")
    texts = [r["paraphrase"] for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to("cuda")
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(struct.pack("<II", len(texts), DIM))
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start+args.batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                emb = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            emb = np.asarray(emb, dtype="<f4", order="C")
            if emb.shape[1] != DIM or not np.isfinite(emb).all():
                raise ValueError(f"bad MedCPT embedding block {emb.shape}")
            emb.tofile(f)
            print(f"embedded={min(start+len(batch),len(texts))}/{len(texts)}", flush=True)

    expected = 8 + len(texts) * DIM * 4
    if args.out.stat().st_size != expected:
        raise RuntimeError("fbin byte-size mismatch")

    manifest = {
        "rows": len(texts),
        "dimension": DIM,
        "dtype": "float32",
        "format": "fbin: <u32 rows><u32 dim><row-major f32>",
        "model": args.model,
        "pooling": "CLS",
        "normalization": "none",
        "metric_target": "inner_product",
        "bytes": args.out.stat().st_size,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
