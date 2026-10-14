#!/usr/bin/env python3
"""Selected NavHints online warm-up study across k=1,2,4,8 historical starts.

Uses the exact same frozen PubMed, DiskANN, directory, PQ, warm-up windows,
and recall grid as qualify_final_core_warmup.py and Figure 8(a).
Only k changes; B=14464, C=2048, 512 regions, persistence disabled.
This run is required before redrawing panel (a): panel (b)'s source is
an independent working-set study and cannot be reused as warm-up curves.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path
from qualify_final_core_warmup import (
    THREADS, BEAM, K, LS, HISTORY, WINDOW, TARGETS, save, slice_fbin,
    slice_gt, rows, mean_match
)

STARTS = (1, 2, 4, 8)


def run(binary, out, tag, q, gt, prefix, ivf, *, history_queries, k=None):
    cfg = {"search_directories": [str(out)], "jobs": [{"type": "disk-index", "content": {
        "source": {"disk-index-source": "Load", "data_type": "float32", "load_path": str(prefix)},
        "search_phase": {
            "queries": str(q), "groundtruth": str(gt), "search_list": list(LS),
            "beam_width": BEAM, "recall_at": K, "num_threads": THREADS,
            "is_flat_search": False, "distance": "inner_product",
            "vector_filters_file": None, "num_nodes_to_cache": None,
            "search_io_limit": None, "post_processor": None
        }
    }}]}
    inp, op, log = (out / f"{tag}.{ext}" for ext in ("input.json", "output.json", "log"))
    save(inp, cfg)
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("DISKANN_"):
            del env[name]
    env["DISKANN_HINT_IVF_FILE"] = str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"] = "8"
    env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    if k is not None:
        env.update({
            "DISKANN_EXPERIENCE_REPLAY": "1",
            "DISKANN_EXPERIENCE_WARMUP": str(history_queries),
            "DISKANN_EXPERIENCE_CACHE_CAPACITY": "2048",
            "DISKANN_EXPERIENCE_HUB_CAPACITY": "0",  # RAM-only Core
            "DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR": "2",
            "DISKANN_EXPERIENCE_FILL_FROM_CACHE": "1",
            "DISKANN_LOCAL_CACHE_MODE": "fifo",
            "DISKANN_LOCAL_RECENT_STARTS": str(k),
        })
    with log.open("w") as stream:
        subprocess.run([str(binary), "run", "--input-file", str(inp),
                        "--output-file", str(op)],
                       stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
    result = sorted(rows(json.loads(op.read_text())), key=lambda r: int(r["search_l"]))
    if [int(x["search_l"]) for x in result] != list(LS):
        raise ValueError(f"{tag}: incomplete or unexpected search-list widths")
    if k is not None:
        stats = [x for x in log.read_text().splitlines() if "EXPERIENCE_STATS " in x]
        if len(stats) != len(LS):
            raise ValueError(f"{tag}: expected {len(LS)} experience stats, saw {len(stats)}")
        for line in stats:
            fields = dict(piece.split("=", 1) for piece in
                          line.split("EXPERIENCE_STATS ", 1)[1].split() if "=" in piece)
            if int(fields.get("cache_capacity", -1)) != 2048 or \
               int(fields.get("hub_capacity", -1)) != 0 or \
               int(fields.get("eval_writes", -1)) != 0:
                raise ValueError(f"{tag}: invalid Core/SSD-write counters: {fields}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "heldout", "gt5000", "index-prefix", "ivf", "work", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    for name in ("binary", "heldout", "gt5000", "index_prefix", "ivf", "work", "out"):
        setattr(a, name, getattr(a, name).resolve())
    for path in (a.binary, a.heldout, a.gt5000, a.ivf):
        if not path.is_file():
            raise FileNotFoundError(path)
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    output = {
        "controller": {
            "junction_budget": 14464, "recent_history": 2048, "regions": 512,
            "starts_tested": list(STARTS), "persistent_exits": False,
        },
        "dataset": "PubMed1M/MedCPT", "windows": [],
        "history_prefixes": list(HISTORY), "measured_window": WINDOW,
        "recall_targets": list(TARGETS),
        "reference": "independent frozen junction-only replay per 500-query suffix",
        "metrics": "mean matched-recall logical SSD provider read savings",
    }
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise ValueError("Not enough available CPU cores")
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            for h in HISTORY:
                if h + WINDOW > 5000:
                    raise ValueError(f"Warm-up window beyond frozen 5K suffix: {h}")
                q_all, gt_all = (a.work / f"h{h}-prefix.{s}"
                                 for s in ("fbin", "gt"))
                q_eval, gt_eval = (a.work / f"h{h}-eval.{s}"
                                   for s in ("fbin", "gt"))
                slice_fbin(a.heldout, q_all, 0, h + WINDOW)
                slice_gt(a.gt5000, gt_all, 0, h + WINDOW)
                slice_fbin(a.heldout, q_eval, h, WINDOW)
                slice_gt(a.gt5000, gt_eval, h, WINDOW)
                if h:
                    q_train, gt_train = (a.work / f"h{h}-history.{s}"
                                         for s in ("fbin", "gt"))
                    slice_fbin(a.heldout, q_train, 0, h)
                    slice_gt(a.gt5000, gt_train, 0, h)
                    run(a.binary, a.out, f"h{h}-ivf-cachewarm",
                        q_train, gt_train, a.index_prefix, a.ivf, history_queries=0)
                baseline = run(a.binary, a.out, f"h{h}-ivf", q_eval, gt_eval,
                               a.index_prefix, a.ivf, history_queries=0)
                result = {"history_queries": h, "measured_queries": WINDOW, "starts": {}}
                for k in STARTS:
                    arm = run(a.binary, a.out, f"h{h}-starts{k}", q_all, gt_all,
                              a.index_prefix, a.ivf, history_queries=h, k=k)
                    match = mean_match(baseline, arm)
                    if match["points"] != len(TARGETS):
                        raise ValueError(
                            f"Missing common Recall@10 anchor at h={h}, k={k}: {match}")
                    result["starts"][str(k)] = match
                output["windows"].append(result)
                save(a.out / "warmup-starters.partial.json", output)
                print("NH_WARMUP_WINDOW " + json.dumps(result), flush=True)
    finally:
        os.sched_setaffinity(0, set(allowed))
    save(a.out / "final-core-warmup-starts.json", output)
    print("NH_FINAL_CORE_WARMUP_STARTS_PASS", flush=True)


if __name__ == "__main__":
    main()
