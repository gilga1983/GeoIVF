#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.qualify_portal_suite import fbin, gtbin, normalize_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as f:
        x = np.asarray(f["train"][:], dtype=np.float32, order="C")
        q = np.asarray(f["test"][:], dtype=np.float32, order="C")
        gt = np.asarray(f["neighbors"][:], dtype=np.int64)
        distance = f.attrs.get("distance", "")
        if isinstance(distance, bytes):
            distance = distance.decode()

    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]:
        raise ValueError("dataset shape mismatch")
    if distance not in ("normalized", "cosine", "angular", "any"):
        raise ValueError(f"unexpected Yahoo distance label: {distance!r}")
    if x.shape[1] % 16 != 0:
        raise ValueError("CatapultDB HEAD requires dimensions divisible by SIMD width 16")

    x = normalize_rows(x)
    q = normalize_rows(q)
    gt = np.asarray(gt[:, :100], dtype=np.int64, order="C")
    if gt.min() < 0 or gt.max() >= len(x):
        raise ValueError("ground truth out of range")

    fbin(args.out / "base.fbin", x)
    fbin(args.out / "queries.fbin", q)
    gtbin(args.out / "groundtruth.bin", gt, x, q)
    np.save(args.out / "queries.npy", q)
    np.save(args.out / "gt_ids.npy", gt.astype(np.uint32, copy=False))

    meta = {
        "dataset": "yahoo-minilm-384-normalized",
        "train_rows": int(x.shape[0]),
        "query_rows": int(q.shape[0]),
        "dimension": int(x.shape[1]),
        "groundtruth_k": int(gt.shape[1]),
        "normalization": "row L2 normalization before squared-L2 search",
        "query_order": "original VIBE test order",
    }
    (args.out / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
