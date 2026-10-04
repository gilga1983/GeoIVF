#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

WORKLOADS = ("cold", "warm")
ARMS = ("lsh-off", "portal-off", "lsh-on", "portal-on")
BEAM = 60
K = 10


def read_ubin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rows, cols = struct.unpack("<II", f.read(8))
        a = np.fromfile(f, dtype="<u4", count=rows * cols)
        if len(a) != rows * cols:
            raise ValueError(f"truncated ubin: {path}")
        if f.read(1):
            raise ValueError(f"extra bytes in ubin: {path}")
    return a.reshape(rows, cols)


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int = K) -> float:
    ids = np.asarray(ids[:, :k], dtype=np.uint32)
    gt = np.asarray(gt[:, :k], dtype=np.uint32)
    if ids.shape != gt.shape:
        raise ValueError(f"shape mismatch {ids.shape} != {gt.shape}")
    hits = sum(len(set(map(int, a)) & set(map(int, b))) for a, b in zip(ids, gt))
    return 100.0 * hits / (len(ids) * k)


def official_qps(path: Path) -> float:
    text = path.read_text()
    m = re.findall(r"Completed\s+\d+\s+searches.*?\(([0-9.]+)\s+QPS\)", text)
    if len(m) != 1:
        raise ValueError(f"expected one QPS in {path}, got {m}")
    return float(m[0])


def read_chunks(path: Path):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "begin\tend\telapsed_seconds\tqps":
        raise ValueError(f"bad chunk file {path}")
    out = []
    for line in lines[1:]:
        begin, end, elapsed, qps = line.split("\t")
        out.append({
            "begin": int(begin),
            "end": int(end),
            "elapsed_seconds": float(elapsed),
            "qps": float(qps),
        })
    return out


def gmean(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.exp(np.mean(np.log(vals))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--replay-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.dir

    gt = {
        "cold": np.load(args.data_dir / "gt_ids.npy"),
        "warm": np.load(args.replay_dir / "gt-z1p0.npy"),
    }

    result = {
        "dataset": "yahoo-minilm-384-normalized",
        "beam": BEAM,
        "k": K,
        "cold_queries": int(len(gt["cold"])),
        "warm_queries": int(len(gt["warm"])),
        "warm_workload": "Zipf alpha 1.0 replay, seed 20261003",
        "portal_policy": {
            "nlist": 1024,
            "nprobe": 32,
            "portal": "nearest database vector to IVF centroid",
        },
        "catapult_policy": {
            "upstream_commit": "7d473050e0a69079d8fc17158f2967486e54380b",
            "num_hash": 10,
            "fifo_capacity_per_source_node": 30,
            "duplicates_allowed": True,
        },
        "results": {},
    }

    for workload in WORKLOADS:
        result["results"][workload] = {}
        for arm in ARMS:
            ids = read_ubin(root / f"{workload}-{arm}.ubin")
            reps = [
                official_qps(root / f"{workload}-{arm}-r{rep}.log")
                for rep in range(3)
            ]
            row = {
                "recall_at_10_percent": recall_at_k(ids, gt[workload]),
                "qps_repetitions": reps,
                "qps_mean": float(np.mean(reps)),
                "qps_std": float(np.std(reps)),
                "chunks": read_chunks(root / f"{workload}-{arm}.chunks.tsv"),
            }
            result["results"][workload][arm] = row

        r = result["results"][workload]
        base = r["lsh-off"]
        comparisons = {
            "portal_only_speedup_vs_lsh_off": r["portal-off"]["qps_mean"] / base["qps_mean"],
            "catapult_only_speedup_vs_lsh_off": r["lsh-on"]["qps_mean"] / base["qps_mean"],
            "combined_speedup_vs_lsh_off": r["portal-on"]["qps_mean"] / base["qps_mean"],
            "catapult_increment_on_portal": r["portal-on"]["qps_mean"] / r["portal-off"]["qps_mean"],
            "portal_increment_on_catapult": r["portal-on"]["qps_mean"] / r["lsh-on"]["qps_mean"],
            "combined_recall_delta_vs_lsh_off":
                r["portal-on"]["recall_at_10_percent"] - base["recall_at_10_percent"],
        }

        if workload == "warm":
            n = min(
                len(r["lsh-off"]["chunks"]),
                len(r["portal-off"]["chunks"]),
                len(r["lsh-on"]["chunks"]),
                len(r["portal-on"]["chunks"]),
            )
            def chunk_ratios(num, den):
                return [
                    r[num]["chunks"][i]["qps"] / r[den]["chunks"][i]["qps"]
                    for i in range(n)
                ]
            combined = chunk_ratios("portal-on", "lsh-off")
            portal = chunk_ratios("portal-off", "lsh-off")
            cache_on_portal = chunk_ratios("portal-on", "portal-off")
            comparisons["combined_chunk_speedup_first_5k_geomean"] = gmean(combined[:5])
            comparisons["combined_chunk_speedup_last_10k_geomean"] = gmean(combined[-10:])
            comparisons["portal_chunk_speedup_first_5k_geomean"] = gmean(portal[:5])
            comparisons["portal_chunk_speedup_last_10k_geomean"] = gmean(portal[-10:])
            comparisons["catapult_on_portal_first_5k_geomean"] = gmean(cache_on_portal[:5])
            comparisons["catapult_on_portal_last_10k_geomean"] = gmean(cache_on_portal[-10:])
            comparisons["combined_chunk_speedups"] = combined

        result["results"][workload]["comparisons"] = comparisons

    router_meta = json.loads((root / "portal-router.json").read_text())
    result["portal_router"] = router_meta
    (root / "catapult-portal-four-arm-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("workload arm qps recall")
    for workload in WORKLOADS:
        for arm in ARMS:
            row = result["results"][workload][arm]
            print(workload, arm, f'{row["qps_mean"]:.2f}', f'{row["recall_at_10_percent"]:.3f}')
        print(workload, json.dumps(result["results"][workload]["comparisons"], sort_keys=True))


if __name__ == "__main__":
    main()
