#!/usr/bin/env python3
"""Selected NavHints: compare junction scores, leaving all else unchanged.

SAME frozen graph, 512-region layout procedure, B=14464, C=2048, k=8,
online FIFO, training/query streams and read accounting. Only the B
junction IDs induced by the score differ. Three rotated real-SSD replays;
each arm is a fresh process using the original frozen DiskANN+NavHints code.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import numpy as np

from qualify_final_core_public import (
    THREADS, BEAM, LS, CFG, save, run, aggregate, interp,
)

MODES = ("sum", "count", "mean")


def check_manifest(root):
    states, bytes_on_disk = {}, set()
    for name in MODES:
        ivf = root / f"ivf-{name}.bin"
        manifest = ivf.with_suffix(ivf.suffix + ".manifest.json")
        if not ivf.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Missing {name} routing state/manifest")
        doc = json.loads(manifest.read_text())
        if (doc.get("landmark_ids") != 14464 or doc.get("nlist") != 512
                or doc.get("state_bytes") != ivf.stat().st_size):
            raise ValueError(f"Score {name}: mismatched final Core memory budget or geometry")
        states[name] = ivf
        bytes_on_disk.add(ivf.stat().st_size)
    if len(bytes_on_disk) != 1:
        raise ValueError("Different accounted routing payloads; comparison not matched")
    return states


def read_manifest(path):
    m = json.loads(path.read_text())
    if m.get("split", {}).get("training_queries", m.get("split", {}).get("static_train_rows")) not in (None, 5000):
        raise ValueError("Incorrect offline training prefix")
    if m["data_type"] not in ("float32", "uint8") or m["metric"] not in ("inner_product", "squared_l2", "l2"):
        raise ValueError("Unexpected distance metric")
    for field in ("heldout5000", "heldout5000_gt"):
        if field not in m["files"] or not Path(m["files"][field]).is_file():
            raise FileNotFoundError(f"Missing frozen {field}")
    return m


def aggregate_matched(summary):
    """Use the current selected paper score as the sole reference recall grid."""
    targets = []
    for L in LS:
        target = float(summary["sum"][str(L)]["recall_percent"])
        costs = {}
        for method in MODES:
            v = interp(summary[method], target, "mean_ios")
            if v is None or v <= 0:
                break
            costs[method] = v
        if len(costs) == len(MODES):
            targets.append({
                "sum_L": L,
                "recall_percent": target,
                "reads": costs,
                "sum_vs_count_saving_percent": 100 * (1 - costs["sum"] / costs["count"]),
                "sum_vs_mean_saving_percent": 100 * (1 - costs["sum"] / costs["mean"]),
            })
    if len(targets) < 3:
        raise ValueError(f"Insufficient jointly reachable recall targets: {len(targets)}")
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "dataset-manifest", "index-prefix", "ivf-root", "work", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    if a.reps != 3:
        raise ValueError("Require three rotated replays for the paper's validation protocol")
    for key in ("binary", "dataset_manifest", "index_prefix", "ivf_root", "work", "out"):
        setattr(a, key, getattr(a, key).resolve())
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.binary.is_file():
        raise FileNotFoundError(a.binary)
    states = check_manifest(a.ivf_root)
    m = read_manifest(a.dataset_manifest)
    queries, gt = (Path(m["files"][key]) for key in ("heldout5000", "heldout5000_gt"))
    if m.get("split", {}).get("warmup_queries", m.get("split", {}).get("online_warmup_queries")) not in (None,4000):
        raise ValueError("Expected 4K online warm-up requests")
    if m.get("split", {}).get("measured_queries") not in (None, 1000):
        raise ValueError("Expected 1K measured requests")
    # The existing NavHints Core runner takes ALL 5000 queries and measures
    # only the last 1000 after causal warm-up, for every search width.
    methods = {name: [] for name in MODES}
    counters = {name: [] for name in MODES}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise ValueError("Insufficient available CPUs for identical four-thread search")
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    device_lock = Path.home() / ".cache/geoivf/speed-device.lock"
    device_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with device_lock.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for repetition in range(a.reps):
                order = MODES[repetition:] + MODES[:repetition]
                for mode in order:
                    rr, st = run(a.binary, a.out,
                                 f"r{repetition}-score-{mode}",
                                 queries, gt, a.index_prefix,
                                 m["data_type"], m["metric"],
                                 states[mode], CFG["core"])
                    if st is None or len(st) != len(LS):
                        raise ValueError(f"Missing per-L runtime statistics for {mode}")
                    for L in LS:
                        row = st[str(L)]
                        if (row.get("cache_capacity") != 2048 or
                            row.get("hub_capacity") != 0 or
                            row.get("eval_writes") != 0 or
                            row.get("writes") != 0):
                            raise ValueError(f"Incorrect final Core / unexpected page write for {mode} L={L}: {row}")
                    methods[mode].append(rr)
                    counters[mode].append(st)
                    point = next(z for z in rr if int(z["search_l"]) == 80)
                    print(f"{m['dataset']} repeat={repetition} mode={mode} "
                          f"L80 recall={point['recall']} reads={point['mean_ios']}",
                          flush=True)
    finally:
        os.sched_setaffinity(0, set(allowed))
    summary = {mode: aggregate(methods[mode]) for mode in MODES}
    matched = aggregate_matched(summary)
    savings = {
        competitor: {
            "targets": len(matched),
            "mean_read_saving_pct": float(np.mean(
                [row[f"sum_vs_{competitor}_saving_percent"] for row in matched])),
            "min_read_saving_pct": min(
                row[f"sum_vs_{competitor}_saving_percent"] for row in matched),
            "max_read_saving_pct": max(
                row[f"sum_vs_{competitor}_saving_percent"] for row in matched),
        } for competitor in ("count", "mean")
    }
    result = {
        "dataset": m["dataset"], "metric": m["metric"],
        "reference": "sum of first expansion positions (unchanged paper design)",
        "other_scores": ["training traversal frequency", "mean first expansion position"],
        "controller": {"B": 14464, "C": 2048, "k": 8,
                       "regions": 512, "persistent_exits": 0},
        "search": {"L": list(LS), "beam": BEAM, "threads": THREADS},
        "protocol": {"repetitions": a.reps, "rotated_orders": True,
                     "offline_train_queries": 5000, "causal_warmup_queries": 4000,
                     "measured_queries": 1000,
                     "comparison": "matched Recall@10, interpolation without extrapolation",
                     "metric_scope": "logical SSD-provider reads per query",
                     "statistical_scope": "same deterministic query population; not independent workload samples"},
        "summary": summary, "matched": matched, "savings_vs_alternative": savings,
    }
    save(a.out / "junction-score-ablation.json", result)
    print("NAVHINTS_JUNCTION_SCORE_ABLATION_PASS", json.dumps(savings, indent=2))


if __name__ == "__main__":
    main()
