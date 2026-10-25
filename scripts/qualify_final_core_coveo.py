#!/usr/bin/env python3
"""Evaluate final B<=14464 / C2048 / k8 NavHints Core and Sample2 on chronological Coveo."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from navhints_io import read_xbin_shape, slice_ground_truth, slice_vectors

THREADS = 4
BEAM = 8
K = 10
LS = (10, 20, 40, 80, 160, 320)
METHODS = ("baseline", "ivf", "core", "full")
# Method keys are frozen result-schema identifiers: ivf=junction-only,
# core=RAM-only NavHints, full=Core plus optional sampled page exits.
CFG = {
    "baseline": {
        "ivf": False,
        "experience": False,
        "recent": 0,
        "hub": 0,
        "sample": 2,
        "fill": False,
    },
    "ivf": {
        "ivf": True,
        "experience": False,
        "recent": 0,
        "hub": 0,
        "sample": 2,
        "fill": False,
    },
    "core": {
        "ivf": True,
        "experience": True,
        "recent": 2048,
        "hub": 0,
        "sample": 2,
        "fill": False,
    },
    "full": {
        "ivf": True,
        "experience": True,
        "recent": 2048,
        "hub": 10,
        "sample": 2,
        "fill": True,
    },
}
STAT_KEYS = (
    "cache_capacity",
    "hub_capacity",
    "sample_denominator",
    "fill",
    "cache_inserts",
    "cache_skips",
    "cache_evictions",
    "active_direct_hubs",
    "persisted_hubs",
    "direct_learned",
    "direct_duplicates",
    "direct_full",
    "sample_trials",
    "sample_accepts",
    "sample_rejects",
    "fifo_evictions",
    "writes",
    "eval_writes",
    "write_slots",
    "write_direct_slots",
    "write_filler_slots",
    "eval_write_slots",
    "eval_write_filler_slots",
    "final_page_slots",
)


def save(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2) + "\n")


def xshape(path: Path) -> tuple[int, int]:
    """Return the original binary shape; retained for replay-helper imports."""
    return read_xbin_shape(path)


def slice_xbin(src, dst, start, count, itemsize=4):
    """Copy the requested replay vectors without changing their element bytes."""
    return slice_vectors(src, dst, start, count, itemsize)


def slice_gt(src, dst, start, count):
    """Copy the frozen ground-truth rows and preserve their block layout."""
    return slice_ground_truth(src, dst, start, count, require_distances=True)


def result_rows(o):
    out = []
    if isinstance(o, dict):
        if "search_l" in o and "mean_latency" in o:
            out.append(o)
        else:
            for v in o.values():
                out.extend(result_rows(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(result_rows(v))
    return out


def parse_stats(log):
    out = {}
    for line in log.read_text().splitlines():
        if "EXPERIENCE_STATS " not in line:
            continue
        kv = {}
        for tok in line.split("EXPERIENCE_STATS ", 1)[1].strip().split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = int(v)
        if "L" in kv:
            out[str(kv["L"])] = {k: kv[k] for k in STAT_KEYS if k in kv}
    if set(out) != set(map(str, LS)):
        raise ValueError(f"missing experience stats: {sorted(out)}")
    return out


def run(binary, out, tag, queries, gt, prefix, ivf, cfg, warmup_rows):
    job = {
        "search_directories": [str(out)],
        "jobs": [
            {
                "type": "disk-index",
                "content": {
                    "source": {
                        "disk-index-source": "Load",
                        "data_type": "float32",
                        "load_path": str(prefix),
                    },
                    "search_phase": {
                        "queries": str(queries),
                        "groundtruth": str(gt),
                        "search_list": list(LS),
                        "beam_width": BEAM,
                        "recall_at": K,
                        "num_threads": THREADS,
                        "is_flat_search": False,
                        "distance": "inner_product",
                        "vector_filters_file": None,
                        "num_nodes_to_cache": None,
                        "search_io_limit": None,
                        "post_processor": None,
                    },
                },
            }
        ],
    }
    inp = out / f"{tag}.input.json"
    op = out / f"{tag}.output.json"
    log = out / f"{tag}.log"
    save(inp, job)
    env = os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):
            env.pop(n, None)
    if cfg["ivf"]:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = "8"
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
    if cfg["experience"]:
        env["DISKANN_EXPERIENCE_REPLAY"] = "1"
        env["DISKANN_EXPERIENCE_WARMUP"] = str(warmup_rows)
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"] = str(cfg["recent"])
        env["DISKANN_EXPERIENCE_HUB_CAPACITY"] = str(cfg["hub"])
        env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"] = str(cfg["sample"])
        env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"] = "1" if cfg["fill"] else "0"
        env["DISKANN_LOCAL_CACHE_MODE"] = "fifo"
        env["DISKANN_LOCAL_RECENT_STARTS"] = "8"
    with log.open("w") as lf:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(op)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
            timeout=3600,
        )
    rr = sorted(
        result_rows(json.loads(op.read_text())), key=lambda r: int(r["search_l"])
    )
    if [int(x["search_l"]) for x in rr] != list(LS):
        raise ValueError(f"{tag}: bad L grid")
    return rr, (parse_stats(log) if cfg["experience"] else None)


def aggregate(reps):
    out = {}
    for search_width in LS:
        rs = [r for rr in reps for r in rr if int(r["search_l"]) == search_width]
        out[str(search_width)] = {
            "recall_percent": float(np.mean([float(r["recall"]) for r in rs])),
            "mean_ios": float(np.mean([float(r["mean_ios"]) for r in rs])),
            "latency_us": float(np.median([float(r["mean_latency"]) for r in rs])),
            "qps": float(np.median([float(r["qps"]) for r in rs])),
            "cpu_us": float(np.median([float(r["mean_cpu_time"]) for r in rs])),
            "comparisons": float(np.mean([float(r["mean_comparisons"]) for r in rs])),
            "hops": float(np.mean([float(r["mean_hops"]) for r in rs])),
        }
    return out


def aggregate_stats(reps, replay_rows, eval_rows):
    out = {}
    for search_width in LS:
        rows = [x[str(search_width)] for x in reps]
        d = {k: float(np.mean([r[k] for r in rows])) for k in STAT_KEYS}
        d["writes_per_query"] = d["writes"] / replay_rows
        d["eval_writes_per_query"] = d["eval_writes"] / eval_rows
        d["acceptance_fraction"] = (
            d["sample_accepts"] / d["sample_trials"] if d["sample_trials"] else 0.0
        )
        d["final_mean_page_occupancy"] = (
            d["final_page_slots"] / d["persisted_hubs"] if d["persisted_hubs"] else 0.0
        )
        out[str(search_width)] = d
    return out


def mono(s):
    """Keep nondecreasing recall points in the frozen search-width order."""
    out = []
    best = -1e99
    for search_width in LS:
        r = s[str(search_width)]["recall_percent"]
        if r + 1e-9 >= best:
            out.append((r, s[str(search_width)]))
            best = max(best, r)
    return out


def interp(s, t, f):
    """Interpolate one recorded cost field without extending recall support."""
    p = mono(s)
    if not p or t < p[0][0] or t > p[-1][0]:
        return None
    for (a, ra), (b, rb) in zip(p, p[1:]):
        if a <= t <= b:
            x = 0 if b <= a + 1e-12 else (t - a) / (b - a)
            return float(ra[f]) + x * (float(rb[f]) - float(ra[f]))
    return float(p[-1][1][f])


def matched(summary, anchor="full"):
    rows = []
    for search_width in LS:
        target = summary[anchor][str(search_width)]["recall_percent"]
        row = {
            "anchor": anchor,
            "anchor_L": search_width,
            "recall_percent": target,
            "comparisons": {},
        }
        for ref in METHODS:
            rio = interp(summary[ref], target, "mean_ios")
            rla = interp(summary[ref], target, "latency_us")
            if rio is None or rla is None:
                row["comparisons"][ref] = {"available": False}
                continue
            aio = summary[anchor][str(search_width)]["mean_ios"]
            ala = summary[anchor][str(search_width)]["latency_us"]
            row["comparisons"][ref] = {
                "available": True,
                "io_saving_percent": 100 * (1 - aio / rio),
                "latency_saving_percent": 100 * (1 - ala / rla),
                "reference_ios": rio,
                "reference_latency_us": rla,
            }
        rows.append(row)
    return rows


def mean_comparison(rows, ref):
    vals = [
        r["comparisons"][ref]
        for r in rows
        if r["comparisons"].get(ref, {}).get("available")
    ]
    return {
        "points": len(vals),
        "mean_io_saving_percent": (
            float(np.mean([v["io_saving_percent"] for v in vals])) if vals else None
        ),
        "mean_latency_saving_percent": (
            float(np.mean([v["latency_saving_percent"] for v in vals]))
            if vals
            else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in (
        "binary",
        "dataset-manifest",
        "replay-gt",
        "index-prefix",
        "ivf",
        "work",
        "out",
    ):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for n in (
        "binary",
        "dataset_manifest",
        "replay_gt",
        "index_prefix",
        "ivf",
        "work",
        "out",
    ):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads(args.dataset_manifest.read_text())
    replay = Path(m["files"]["replay"])
    warmup = int(m["split"]["online_warmup_rows"])
    eval_rows = int(m["split"]["measured_rows"])
    replay_rows = warmup + eval_rows
    if xshape(replay)[0] != replay_rows:
        raise ValueError("replay rows do not match manifest")
    gt_rows, _ = read_xbin_shape(args.replay_gt)
    if gt_rows != replay_rows:
        raise ValueError("GT rows do not match replay")

    evalq = args.work / "eval.fbin"
    evalgt = args.work / "eval.gt"
    warmq = args.work / "warm.fbin"
    warmgt = args.work / "warm.gt"
    slice_xbin(replay, warmq, 0, warmup)
    slice_gt(args.replay_gt, warmgt, 0, warmup)
    slice_xbin(replay, evalq, warmup, eval_rows)
    slice_gt(args.replay_gt, evalgt, warmup, eval_rows)

    runs = {x: [] for x in METHODS}
    stats = {x: [] for x in METHODS if CFG[x]["experience"]}
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lockp = Path.home() / ".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(METHODS)
                order = list(METHODS[shift:] + METHODS[:shift])
                print(f"rep={rep} order={' '.join(order)}", flush=True)
                for method in order:
                    cfg = CFG[method]
                    if cfg["experience"]:
                        q, gt = replay, args.replay_gt
                    else:
                        # Equalize storage warmth using the same real-demand prefix.
                        run(
                            args.binary,
                            args.out,
                            f"r{rep}-{method}-warm",
                            warmq,
                            warmgt,
                            args.index_prefix,
                            args.ivf,
                            cfg,
                            0,
                        )
                        q, gt = evalq, evalgt
                    rr, st = run(
                        args.binary,
                        args.out,
                        f"r{rep}-{method}",
                        q,
                        gt,
                        args.index_prefix,
                        args.ivf,
                        cfg,
                        warmup,
                    )
                    runs[method].append(rr)
                    if st is not None:
                        stats[method].append(st)
                    r80 = next(x for x in rr if int(x["search_l"]) == 80)
                    print(
                        f"{method} L80 recall={float(r80['recall']):.3f} io={float(r80['mean_ios']):.2f} lat={float(r80['mean_latency']):.1f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0, set(allowed))

    summary = {mth: aggregate(runs[mth]) for mth in METHODS}
    write_stats = {
        mth: aggregate_stats(stats[mth], replay_rows, eval_rows) for mth in stats
    }
    sample_match = matched(summary, "full")
    recent_match = matched(summary, "core")
    ivfm = json.loads(
        args.ivf.with_suffix(args.ivf.suffix + ".manifest.json").read_text()
    )

    result = {
        "dataset": m["dataset"],
        "source": m["source"],
        "split": m["split"],
        "controller": {
            "entry_ids": int(ivfm["landmark_ids"]),
            "nlist": int(ivfm["nlist"]),
            "nprobe": 8,
            "recent_result_hints": 2048,
            "extra_ranked_starts": 8,
            "nominal_junction_budget": 14464,
            "hub_winners": 10,
            "sample_denominator": 2,
        },
        "search": {"K": K, "Ls": list(LS), "beam": BEAM, "threads": THREADS},
        "summary": summary,
        "write_stats": write_stats,
        "matched_sample2": sample_match,
        "matched_core": recent_match,
        "mean_sample2_vs": {
            ref: mean_comparison(sample_match, ref)
            for ref in ("baseline", "ivf", "core")
        },
        "mean_core_vs": {
            ref: mean_comparison(recent_match, ref) for ref in ("baseline", "ivf")
        },
    }
    save(args.out / "coveo-final-core.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
