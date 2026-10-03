#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np


def fbin(path: Path, array):
    a = np.asarray(array, dtype=np.float32, order="C")
    with Path(path).open("wb") as f:
        np.asarray(a.shape, dtype="<u4").tofile(f)
        for s in range(0, len(a), 32768):
            np.asarray(a[s:s+32768], dtype="<f4", order="C").tofile(f)


def gtbin(path: Path, ids, x, q):
    ids = np.asarray(ids, dtype="<u4", order="C")
    with Path(path).open("wb") as f:
        np.asarray(ids.shape, dtype="<u4").tofile(f)
        ids.tofile(f)
        for s in range(0, len(ids), 512):
            block = ids[s:s+512]
            qq = q[s:s+512].astype(np.float64)
            d = np.sum((x[block].astype(np.float64) - qq[:, None, :])**2, axis=2).astype("<f4")
            d.tofile(f)


def normalize_rows(a):
    a = np.asarray(a, dtype=np.float32, order="C")
    norms = np.linalg.norm(a.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("zero/nonfinite vector")
    return np.asarray(a / norms[:, None], dtype=np.float32, order="C")


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
