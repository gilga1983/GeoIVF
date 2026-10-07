#!/usr/bin/env python3
"""Periodically persist ten independent winners drawn from the EXISTING value cache.

This isolates one update policy, without adding another cache:
  * every completed search updates the original online 512-ID SkipDup cache;
  * at every Xth query, choose one completed rank-1 winner plus up to nine
    highest-PQ-ranked distinct successful IDs from that cache;
  * write the resulting ten-ID subset to that query's selected 16K hub;
  * the hub metadata becomes available only to subsequent searches, and only
    when its page is read.

Physical page writes are counted but not executed in this initial sweep.
All treatments replay identical chronological 5K queries, evaluate the final
1K, and charge real PQ lookup / online RAM update time.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from qualify_online_combined import (
    LS, BEAM, K,
    save, range_fbin, result_rows, aggregate, mono, interp,
)

METHODS = (
    "cache512", "perhub10", "write1", "write10", "write25", "write50",
)

WRITE_RE = re.compile(
    r"CACHE_TO_HUB_WRITES L=(\d+) cadence=(\d+) slots=(\d+) "
    r"logical_page_writes=(\d+) unchanged=(\d+) new_pages=(\d+) "
    r"ids_written=(\d+) full10_writes=(\d+) short_writes=(\d+) "
    r"unique_hub_pages=(\d+) writes_per_query=([0-9.]+)"
)
CACHE_RE = re.compile(
    r"ONLINE_VALUE_STATS L=(\d+) capacity=(\d+) occupancy=(\d+) "
    r"inserts=(\d+) duplicate_skips=(\d+) evictions=(\d+)"
)


def run(binary, out, tag, q, gt, index_prefix, ivf, hub, cache, cadence):
    cfg = {
        "search_directories": [str(out)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": str(index_prefix),
                },
                "search_phase": {
                    "queries": str(q),
                    "groundtruth": str(gt),
                    "search_list": list(LS),
                    "beam_width": BEAM,
                    "recall_at": K,
                    "num_threads": 4,
                    "is_flat_search": False,
                    "distance": "inner_product",
                    "vector_filters_file": None,
                    "num_nodes_to_cache": None,
                    "search_io_limit": None,
                    "post_processor": None,
                },
            },
        }],
    }
    inp, output, log = (
        out / f"{tag}.input.json",
        out / f"{tag}.output.json",
        out / f"{tag}.log",
    )
    save(inp, cfg)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("DISKANN_"):
            env.pop(name, None)
    env.update({
        "DISKANN_HINT_IVF_FILE": str(ivf),
        "DISKANN_HINT_IVF_NPROBE": "8",
        "DISKANN_HINT_IVF_MAX_STARTS": "1",
        "DISKANN_VERTEX_HINT_VARIANT": "1",
        "DISKANN_ONLINE_HUB_REPLAY": "1",
        "DISKANN_ONLINE_HUB_CAPACITY": str(hub),
        "DISKANN_ONLINE_HUB_WARMUP": "4000",
        "DISKANN_ONLINE_HUB_FLUSH_BATCH": "10",
        "DISKANN_COMBINED_VALUE_CACHE_CAPACITY": str(cache),
        "DISKANN_CACHE_TO_HUB_WRITE_EVERY": str(cadence),
        "DISKANN_CACHE_TO_HUB_WRITE_WINNERS": "10",
    })
    with log.open("w") as f:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp),
             "--output-file", str(output)],
            stdout=f, stderr=subprocess.STDOUT, env=env, check=True,
        )

    rows = sorted(result_rows(json.loads(output.read_text())),
                  key=lambda r: int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS):
        raise ValueError(f"{tag}: unexpected L grid")
    txt = log.read_text()
    writes = {}
    cache_info = {}
    for m in WRITE_RE.finditer(txt):
        l = str(m.group(1))
        writes[l] = {
            "cadence": int(m.group(2)),
            "slots": int(m.group(3)),
            "logical_page_writes": int(m.group(4)),
            "unchanged": int(m.group(5)),
            "new_pages": int(m.group(6)),
            "ids_written": int(m.group(7)),
            "full10_writes": int(m.group(8)),
            "short_writes": int(m.group(9)),
            "unique_hub_pages": int(m.group(10)),
            "writes_per_query": float(m.group(11)),
        }
    for m in CACHE_RE.finditer(txt):
        cache_info[str(m.group(1))] = {
            "capacity": int(m.group(2)),
            "occupancy": int(m.group(3)),
            "inserts": int(m.group(4)),
            "duplicate_skips": int(m.group(5)),
            "evictions": int(m.group(6)),
        }
    if len(writes) != len(LS) or len(cache_info) != len(LS):
        raise ValueError(f"{tag}: missing live write/cache stats")
    for row in writes.values():
        if row["cadence"] > 0 and row["new_pages"] > row["logical_page_writes"]:
            raise ValueError("new page count exceeds writes")
        if row["full10_writes"] + row["short_writes"] != row["logical_page_writes"]:
            raise ValueError("inconsistent full-ten write counts")
        if row["ids_written"] > row["logical_page_writes"] * 10:
            raise ValueError("wrote too many candidate IDs")
    return rows, {"writes": writes, "cache": cache_info}


def mean_diag(reps):
    result = {}
    for part in ("writes", "cache"):
        result[part] = {}
        for l in LS:
            values = [x[part][str(l)] for x in reps]
            result[part][str(l)] = {
                key: float(np.mean([v[key] for v in values]))
                for key in values[0]
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    for n in ("binary", "queries", "gt5000", "index-prefix",
              "ivf-16k", "work", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    for n in ("binary", "queries", "gt5000", "index_prefix",
              "ivf_16k", "work", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    q = args.work / "replay5000.fbin"
    range_fbin(args.queries, q, 5000, 5000)

    configs = {
        "cache512": (0, 512, 0),
        "perhub10": (10, 512, 0),
        "write1": (10, 512, 1),
        "write10": (10, 512, 10),
        "write25": (10, 512, 25),
        "write50": (10, 512, 50),
    }
    runs = {m: [] for m in METHODS}
    diagnostics = {m: [] for m in METHODS}
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:4]))
    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(METHODS)
                order = list(METHODS[shift:] + METHODS[:shift])
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for name in order:
                    hub, cache, cadence = configs[name]
                    rr, diag = run(
                        args.binary, args.out, f"r{rep}-{name}", q,
                        args.gt5000, args.index_prefix, args.ivf_16k,
                        hub, cache, cadence,
                    )
                    runs[name].append(rr)
                    diagnostics[name].append(diag)
                    r40 = next(x for x in rr if int(x["search_l"]) == 40)
                    writes40 = diag["writes"]["40"]
                    print(
                        f"{name}: L40 recall={float(r40['recall']):.3f}, "
                        f"mean_io={float(r40['mean_ios']):.3f}, "
                        f"lat_us={float(r40['mean_latency']):.1f}, "
                        f"writes={writes40['logical_page_writes']}, "
                        f"full10={writes40['full10_writes']}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {name: aggregate(runs[name]) for name in METHODS}
    diag = {name: mean_diag(diagnostics[name]) for name in METHODS}
    targets = (35., 45., 55., 65., 72., 75.)
    fixed = {}
    for target in targets:
        base_io = interp(summary["cache512"], target, "mean_ios")
        base_lat = interp(summary["cache512"], target, "latency_us")
        if base_io is None:
            continue
        fixed[str(target)] = {}
        for name in METHODS:
            io = interp(summary[name], target, "mean_ios")
            lat = interp(summary[name], target, "latency_us")
            if io is None or lat is None:
                continue
            fixed[str(target)][name] = {
                "io_change_percent_over_cache512": 100 * (io / base_io - 1),
                "latency_change_percent_over_cache512": 100 * (lat / base_lat - 1),
            }

    rank = []
    for name in METHODS[1:]:
        usable = [fixed[str(t)][name] for t in targets
                  if str(t) in fixed and name in fixed[str(t)]]
        if not usable:
            continue
        rank.append({
            "method": name,
            "mean_incremental_io_change_percent_vs_cache512": float(np.mean(
                [v["io_change_percent_over_cache512"] for v in usable])),
            "mean_incremental_latency_change_percent_vs_cache512": float(np.mean(
                [v["latency_change_percent_over_cache512"] for v in usable])),
            "mean_writes_per_query": diag[name]["writes"]["160"]["writes_per_query"],
            "mean_full10_writes": diag[name]["writes"]["160"]["full10_writes"],
        })
    rank.sort(key=lambda x: x["mean_incremental_io_change_percent_vs_cache512"])

    result = {
        "question": "can periodic writes of 10 query-relevant winners from the existing 512-ID cache replace per-hub winner accumulation?",
        "mechanism": "one completed rank1 winner + top9 other PQ-ranked cached winners, distinct; one hub page write every X queries",
        "online_state": "cold start; query-selected hub; no additional write cache",
        "disk_writes": "logical page mutations modeled; physical I/O not yet charged",
        "evaluation": {
            "reps": args.reps, "Ls": list(LS),
            "warm_requests": 4000, "measured_requests": 1000,
        },
        "summary": summary,
        "diagnostics": diag,
        "fixed_recall": fixed,
        "ranking": rank,
    }
    save(args.out / "cache-to-hub.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
