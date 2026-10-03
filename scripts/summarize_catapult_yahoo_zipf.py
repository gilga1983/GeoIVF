#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np


ALPHAS = ("0p0", "0p75", "1p0", "1p25")
BEAMS = (20, 60, 100)
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
    hits = sum(len(set(map(int, a)) & set(map(int, b))) for a, b in zip(ids, gt))
    return 100.0 * hits / (len(ids) * k)


def official_qps(path: Path) -> float:
    text = path.read_text()
    m = re.findall(r"Completed\s+\d+\s+searches.*?\(([0-9.]+)\s+QPS\)", text)
    if len(m) != 1:
        raise ValueError(f"expected one QPS in {path}, got {m}")
    return float(m[0])


def wrapper_qps(path: Path) -> float:
    vals = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v
    return float(vals["qps"])


def read_chunks(path: Path):
    rows = []
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "begin\tend\telapsed_seconds\tqps":
        raise ValueError(f"bad chunk file: {path}")
    for line in lines[1:]:
        begin, end, elapsed, qps = line.split("\t")
        rows.append({
            "begin": int(begin),
            "end": int(end),
            "elapsed_seconds": float(elapsed),
            "qps": float(qps),
        })
    return rows


def geom_mean_speedup(on, off):
    ratios = np.asarray(on, dtype=np.float64) / np.asarray(off, dtype=np.float64)
    return float(np.exp(np.mean(np.log(ratios))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--replay-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.dir
    replay = args.replay_dir

    replay_meta = json.loads((replay / "zipf-replay.json").read_text())
    result = {
        "dataset": "yahoo-minilm-384-normalized",
        "replay": replay_meta,
        "beams": list(BEAMS),
        "catapult_official_commit": "7d473050e0a69079d8fc17158f2967486e54380b",
        "catapult_head_policy": {
            "num_hash": 10,
            "catapult_capacity": 30,
            "engine_seed": 42,
        },
        "results": {},
    }

    for alpha in ALPHAS:
        gt = np.load(replay / f"gt-z{alpha}.npy")
        result["results"][alpha] = {}
        for beam in BEAMS:
            row = {}
            chunks = {}
            for arm in ARMS:
                ids = read_ubin(root / f"zipf-z{alpha}-{arm}-b{beam}.ubin")
                qps_reps = [
                    official_qps(root / f"zipf-z{alpha}-{arm}-b{beam}-r{rep}.log")
                    for rep in range(3)
                ]
                ch = read_chunks(root / f"zipf-z{alpha}-{arm}-b{beam}.chunks.tsv")
                chunks[arm] = ch
                row[arm] = {
                    "recall_at_10_percent": recall_at_k(ids, gt, 10),
                    "official_qps_repetitions": qps_reps,
                    "official_qps_mean": float(np.mean(qps_reps)),
                    "official_qps_std": float(np.std(qps_reps)),
                    "wrapper_qps": wrapper_qps(root / f"zipf-z{alpha}-{arm}-b{beam}.txt"),
                    "chunks": ch,
                }

            if len(chunks["on"]) != len(chunks["off"]):
                raise ValueError("chunk count mismatch")
            speedups = [
                a["qps"] / b["qps"]
                for a, b in zip(chunks["on"], chunks["off"])
            ]
            first5 = speedups[: min(5, len(speedups))]
            last10 = speedups[max(0, len(speedups) - 10):]
            row["comparison"] = {
                "official_qps_speedup_on_over_off":
                    row["on"]["official_qps_mean"] / row["off"]["official_qps_mean"],
                "recall_delta_points_on_minus_off":
                    row["on"]["recall_at_10_percent"] - row["off"]["recall_at_10_percent"],
                "wrapper_chunk_speedups": speedups,
                "first_5k_geomean_speedup": geom_mean_speedup(
                    [x["qps"] for x in chunks["on"][:min(5, len(speedups))]],
                    [x["qps"] for x in chunks["off"][:min(5, len(speedups))]],
                ),
                "last_10k_geomean_speedup": geom_mean_speedup(
                    [x["qps"] for x in chunks["on"][max(0, len(speedups)-10):]],
                    [x["qps"] for x in chunks["off"][max(0, len(speedups)-10):]],
                ),
                "chunk_speedup_min": float(np.min(speedups)),
                "chunk_speedup_max": float(np.max(speedups)),
            }
            result["results"][alpha][str(beam)] = row

    (root / "catapult-yahoo-zipf-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("alpha beam off_qps on_qps speedup off_recall on_recall first5k last10k")
    for alpha in ALPHAS:
        for beam in BEAMS:
            r = result["results"][alpha][str(beam)]
            c = r["comparison"]
            print(
                alpha, beam,
                f'{r["off"]["official_qps_mean"]:.2f}',
                f'{r["on"]["official_qps_mean"]:.2f}',
                f'{c["official_qps_speedup_on_over_off"]:.4f}',
                f'{r["off"]["recall_at_10_percent"]:.3f}',
                f'{r["on"]["recall_at_10_percent"]:.3f}',
                f'{c["first_5k_geomean_speedup"]:.4f}',
                f'{c["last_10k_geomean_speedup"]:.4f}',
            )


if __name__ == "__main__":
    main()
