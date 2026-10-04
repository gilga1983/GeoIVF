#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from geoivf.index import train_faiss
from scripts.qualify_portal_suite import NLIST, build_portal_table, normalize_rows
from scripts.qualify_portal_integrated import write_router


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as f:
        x = np.asarray(f["train"][:], dtype=np.float32, order="C")
        distance = f.attrs.get("distance", "")
        if isinstance(distance, bytes):
            distance = distance.decode()
    if distance not in ("normalized", "any", "cosine", "angular"):
        raise ValueError(f"unexpected Yahoo distance label: {distance!r}")
    x = normalize_rows(x)

    started = time.perf_counter()
    centers, labels = train_faiss(x, NLIST, 12345, min(100000, len(x)))
    train_s = time.perf_counter() - started
    started = time.perf_counter()
    portal_ids, portal_vecs = build_portal_table(x, centers, labels)
    portal_s = time.perf_counter() - started
    write_router(args.out, centers, portal_ids, portal_vecs)

    meta = {
        "dataset": "yahoo-minilm-384-normalized",
        "nlist": NLIST,
        "dimension": int(x.shape[1]),
        "training_seed": 12345,
        "training_rows": min(100000, len(x)),
        "portal_policy": "nearest database vector to IVF centroid",
        "router_bytes": args.out.stat().st_size,
        "router_mib": args.out.stat().st_size / (1 << 20),
        "kmeans_seconds": train_s,
        "portal_selection_seconds": portal_s,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
