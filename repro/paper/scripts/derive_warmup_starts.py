#!/usr/bin/env python3
"""Reproduce Figure 8(a)'s four NavHints start-count curves from frozen SSD evidence.

Required inputs:
  - GeoIVF run 38516776145 artifact 11873220542, a ZIP containing
    warmup-starts/final-core-warmup-starts.json and arm-level logs.
  - The original published k=8 CSV, eval_warmup_final_core.csv.

Fails before writing output if any experiment-state, recall-coverage,
physical-write or original k=8 equivalence check fails. Data use
six common Recall@10 targets per measured 500-query window.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

SOURCE_SHA = "b74857e418746498249955dcc432d6bd99e5cf07"
EXPERIMENT_ID = "38516776145"
HISTORY = (0, 250, 500, 1000, 2000, 3000, 4000)
K_VALUES = (1, 2, 4, 8)
SEARCH_WIDTHS = (10, 20, 40, 80, 160, 320)
RECALL_TARGETS = (35, 45, 55, 65, 72, 75)


def parse_stats(log: str) -> dict[int, dict[str, int]]:
    found: dict[int, dict[str, int]] = {}
    for line in log.splitlines():
        if "EXPERIENCE_STATS " not in line:
            continue
        kv = {}
        for chunk in line.split("EXPERIENCE_STATS ", 1)[1].split():
            if "=" in chunk:
                field, value = chunk.split("=", 1)
                try:
                    kv[field] = int(value)
                except ValueError:
                    continue
        if "L" not in kv or kv["L"] in found:
            raise ValueError("Missing or duplicated L in experiment statistics")
        found[kv["L"]] = kv
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--existing-k8", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with zipfile.ZipFile(a.artifact) as archive:
        source_sha = archive.read("geoivf-commit.txt").decode().strip()
        if source_sha != SOURCE_SHA:
            raise ValueError(f"Expected GeoIVF {SOURCE_SHA}, found {source_sha}")
        if archive.read("status.txt").decode().strip() != "NH_FINAL_CORE_WARMUP_STARTS_SUCCESS":
            raise ValueError("Measured run lacks its successful completion marker")
        contract = archive.read("controller-contract.txt").decode().strip()
        if "B=14464 C=2048 nlist=512 k=1,2,4,8 persistence=OFF" not in contract:
            raise ValueError("Expected final RAM-only NavHints experimental contract")
        data = json.loads(archive.read("warmup-starts/final-core-warmup-starts.json"))
        if data.get("controller") != {
            "junction_budget": 14464,
            "recent_history": 2048,
            "regions": 512,
            "starts_tested": [1, 2, 4, 8],
            "persistent_exits": False,
        }:
            raise ValueError("Warm-up artifact is not selected NavHints")
        if tuple(data.get("history_prefixes", [])) != HISTORY:
            raise ValueError("Unexpected history-prefix grid")
        if tuple(data.get("recall_targets", [])) != RECALL_TARGETS:
            raise ValueError("Unexpected matched-recall grid")
        if data.get("measured_window") != 500:
            raise ValueError("Wrong measured suffix length")
        windows = data["windows"]
        if [r["history_queries"] for r in windows] != list(HISTORY):
            raise ValueError("Missing or reordered warm-up windows")
        rows = []
        for row in windows:
            h = int(row["history_queries"])
            if row["measured_queries"] != 500:
                raise ValueError("Measured suffix must be 500 queries")
            if set(row["starts"]) != set(map(str, K_VALUES)):
                raise ValueError(f"Missing or additional start-count variants at h={h}")
            point = {"history": str(h)}
            for k in K_VALUES:
                item = row["starts"][str(k)]
                if item["points"] != len(RECALL_TARGETS):
                    raise ValueError(f"h={h}, k={k}: not all six recall anchors are supported")
                v = float(item["mean_io_saving_percent"])
                if not math.isfinite(v) or abs(v) > 100:
                    raise ValueError(f"h={h}, k={k}: invalid SSD read saving")
                log = archive.read(f"warmup-starts/h{h}-starts{k}.log").decode()
                per_width = parse_stats(log)
                if tuple(sorted(per_width)) != SEARCH_WIDTHS:
                    raise ValueError(f"h={h}, k={k}: missing search width statistics")
                for L in SEARCH_WIDTHS:
                    values = per_width[L]
                    if any(values.get(f, -1) != goal for f, goal in [
                        ("cache_capacity", 2048), ("hub_capacity", 0), ("eval_writes", 0)
                    ]):
                        raise ValueError(f"h={h}, k={k}, L={L}: wrong state or unexpected SSD writes")
                point[f"k{k}"] = repr(v)
            rows.append(point)
    with a.existing_k8.open(newline="") as f:
        previous = list(csv.DictReader(f))
    if len(previous) != len(rows):
        raise ValueError("Original k=8 history grid is not seven points")
    for old, new in zip(previous, rows):
        if int(old["history"]) != int(new["history"]):
            raise ValueError("Warm-up history grid changed")
        if not math.isclose(float(old["core"]), float(new["k8"]),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Published k=8 changed at h={new['history']}")
    # Do not touch the original k=8 file. Emit one file with all four curves.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["history", "k1", "k2", "k4", "k8"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("NH_FIGURE8_WARMUP_STARTS_VALIDATED", EXPERIMENT_ID)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
