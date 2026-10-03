#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np


BEAMS = (20, 40, 60, 100, 200)
ARMS = ("off", "on")


def read_ubin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rows, cols = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows * cols)
        if len(a) != rows * cols:
            raise ValueError(f"truncated ubin: {path}")
        if f.read(1):
            raise ValueError(f"extra bytes in ubin: {path}")
    return a.reshape(rows, cols)


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    ids = np.asarray(ids[:, :k], dtype=np.uint32)
    gt = np.asarray(gt[:, :k], dtype=np.uint32)
    if ids.shape != gt.shape:
        raise ValueError(f"shape mismatch ids={ids.shape} gt={gt.shape}")
    hits = 0
    for a, b in zip(ids, gt):
        hits += len(set(map(int, a)) & set(map(int, b)))
    return 100.0 * hits / (len(ids) * k)


def official_qps(log: Path) -> float:
    text = log.read_text()
    m = re.findall(r"Completed\s+\d+\s+searches.*?\(([0-9.]+)\s+QPS\)", text)
    if len(m) != 1:
        raise ValueError(f"expected one QPS in {log}, got {m}")
    return float(m[0])


def parse_diskannpy(path: Path):
    obj = json.loads(path.read_text())
    out = {}
    for c in BEAMS:
        row = obj["search"][str(c)]
        out[str(c)] = {
            "qps_mean": float(row["qps_mean"]),
            "qps_std": float(row["qps_std"]),
            "recall_at_10_percent": float(row["recall_at_10_percent"]),
            "repetitions": row["repetitions"],
        }
    return {
        "diskannpy_version": obj["diskannpy_version"],
        "build_seconds": obj["build_seconds"],
        "graph_bytes": obj["graph_bytes"],
        "payload_bytes": obj["payload_bytes"],
        "build": obj["build"],
        "search": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.dir

    gt = np.load(root / "gt_ids.npy")
    result = {
        "dataset": "yahoo-minilm-384-normalized",
        "queries": int(len(gt)),
        "k": 10,
        "diskann_baseline": "official diskannpy 0.7.0 StaticMemoryIndex",
        "catapult_official_commit": "7d473050e0a69079d8fc17158f2967486e54380b",
        "catapult_head_policy": {
            "num_hash": 10,
            "catapult_capacity": 30,
            "engine_seed": 42,
        },
        "diskann_medoid": parse_diskannpy(root / "diskannpy-result.json"),
        "catapult_official": {},
    }

    for arm in ARMS:
        result["catapult_official"][arm] = {}
        for beam in BEAMS:
            ids = read_ubin(root / f"catapult-{arm}-b{beam}.ubin")
            qps_values = [
                official_qps(root / f"catapult-{arm}-b{beam}-r{rep}.log")
                for rep in range(3)
            ]
            wrapper = {}
            for line in (root / f"catapult-{arm}-b{beam}.txt").read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    wrapper[k] = v
            result["catapult_official"][arm][str(beam)] = {
                "recall_at_10_percent": recall_at_k(ids, gt, 10),
                "official_cli_qps_repetitions": qps_values,
                "official_cli_qps_mean": float(np.mean(qps_values)),
                "official_cli_qps_std": float(np.std(qps_values)),
                "measurement_wrapper_qps": float(wrapper["qps"]),
            }

    (root / "catapult-official-yahoo-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
