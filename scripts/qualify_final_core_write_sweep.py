#!/usr/bin/env python3
"""Final-Core sampled-exit read/write frontier on the frozen PubMed SSD index.

The chosen Core is B14464,C2048,k8. This runs Sample1/2/4/10 and Sample2
with/without empty-slot seeding at the SAME SSD/index, heldout queries and
training prefix. Three repeated rotated methods, warm=4000, eval=1000.
The write counts are logical controller requests, not physical amplification.
"""
import argparse, fcntl, json, os, statistics
from pathlib import Path
from qualify_final_core_public import (
    K,
    BEAM,
    THREADS,
    LS,
    run,
    aggregate,
    aggregate_stats,
    slice_xbin,
    slice_gt,
    interp,
    save,
    xshape,
)

METHODS = ("ivf", "core", "sample1", "sample2", "sample4", "sample10", "nofill2")
CFG = {
    "ivf": {
        "ivf": True,
        "experience": False,
        "core": 0,
        "hub": 0,
        "sample": 2,
        "fill": False,
    },
    "core": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 0,
        "sample": 2,
        "fill": False,
    },
    "sample1": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 10,
        "sample": 1,
        "fill": True,
    },
    "sample2": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 10,
        "sample": 2,
        "fill": True,
    },
    "sample4": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 10,
        "sample": 4,
        "fill": True,
    },
    "sample10": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 10,
        "sample": 10,
        "fill": True,
    },
    "nofill2": {
        "ivf": True,
        "experience": True,
        "core": 2048,
        "hub": 10,
        "sample": 2,
        "fill": False,
    },
}


def common(summary, anchor="sample2"):
    anchors = []
    for L in LS:
        target = summary[anchor][str(L)]["recall_percent"]
        vals = {}
        for mode in METHODS:
            cost = interp(summary[mode], target, "mean_ios")
            lat = interp(summary[mode], target, "latency_us")
            if cost is None or lat is None:
                break
            vals[mode] = {"reads": cost, "latency_us": lat}
        if len(vals) != len(METHODS):
            continue
        b = vals["ivf"]["reads"]
        if b <= 0:
            raise ValueError("bad base reads")
        out = {
            "anchor_L": L,
            "recall_percent": target,
            "comparison": vals,
            "savings_vs_junction": {
                k: 100 * (1 - v["reads"] / b) for k, v in vals.items()
            },
            "extra_savings_over_core": {
                k: 100 * (1 - v["reads"] / vals["core"]["reads"])
                for k, v in vals.items()
            },
        }
        anchors.append(out)
    if len(anchors) < 3:
        raise RuntimeError(f"only {len(anchors)} common recall anchors")
    return anchors


def main():
    ap = argparse.ArgumentParser()
    for key in ("binary", "queries", "gt5000", "index-prefix", "ivf", "work", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for k in ("binary", "queries", "gt5000", "index_prefix", "ivf", "work", "out"):
        setattr(args, k, getattr(args, k).resolve())
    if args.reps != 3 or xshape(args.queries) != (5000, 768):
        raise ValueError("require three reps and original heldout 5000 PubMed x 768")
    for file in (args.binary, args.queries, args.gt5000, args.ivf):
        if not file.is_file():
            raise FileNotFoundError(file)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    qwarm = args.work / "warm4000.fbin"
    qw = args.work / "eval1000.fbin"
    gw = args.work / "warm4000.gt"
    gt = args.work / "eval1000.gt"
    slice_xbin(args.queries, qwarm, 0, 4000, 4)
    slice_xbin(args.queries, qw, 4000, 1000, 4)
    slice_gt(args.gt5000, gw, 0, 4000)
    slice_gt(args.gt5000, gt, 4000, 1000)
    original = sorted(os.sched_getaffinity(0))
    if len(original) < THREADS:
        raise RuntimeError("self-hosted runner requires four cores")
    os.sched_setaffinity(0, set(original[:THREADS]))
    measurements = {m: [] for m in METHODS}
    metrics = {m: [] for m in METHODS if CFG[m]["experience"]}
    lock = Path.home() / ".cache/geoivf/speed-device.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("w") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            for rep in range(args.reps):
                i = rep % len(METHODS)
                order = list(METHODS[i:] + METHODS[:i])
                for mode in order:
                    cfg = CFG[mode]
                    if not cfg["experience"]:
                        run(
                            args.binary,
                            args.out,
                            f"r{rep}-{mode}-cachewarm",
                            qwarm,
                            gw,
                            args.index_prefix,
                            "float32",
                            "inner_product",
                            args.ivf,
                            cfg,
                        )
                    query = args.queries if cfg["experience"] else qw
                    truth = args.gt5000 if cfg["experience"] else gt
                    observed, stats = run(
                        args.binary,
                        args.out,
                        f"r{rep}-{mode}",
                        query,
                        truth,
                        args.index_prefix,
                        "float32",
                        "inner_product",
                        args.ivf,
                        cfg,
                    )
                    measurements[mode].append(observed)
                    if stats is not None:
                        metrics[mode].append(stats)
                    save(args.out / "partial-method-results.json", measurements)
                    print(f"WRITE_SWEEP rep={rep} mode={mode} completed", flush=True)
    finally:
        os.sched_setaffinity(0, set(original))
    summary = {m: aggregate(measurements[m]) for m in METHODS}
    st = {m: aggregate_stats(metrics[m]) for m in metrics}
    anchors = common(summary)
    vals = {}
    for m in METHODS:
        vals[m] = {
            "matched_anchor_count": len(anchors),
            "reads_saved_vs_junction_percent": statistics.mean(
                row["savings_vs_junction"][m] for row in anchors
            ),
            "reads_saved_vs_core_percent": statistics.mean(
                row["extra_savings_over_core"][m] for row in anchors
            ),
            "measured_L160_logical_writes_per_query": (
                st[m]["160"]["eval_writes_per_query"] if m in st else 0.0
            ),
        }
    results = {
        "dataset": "PubMed1M/MedCPT",
        "controller": {
            "B": 14464,
            "C": 2048,
            "k": 8,
            "hub_winners": 10,
            "sampling": list(CFG),
            "source": "causal completion-order feedback",
        },
        "query_protocol": {
            "static_train": 5000,
            "warmup": 4000,
            "measured": 1000,
            "repetitions": args.reps,
            "Ls": list(LS),
        },
        "methods": CFG,
        "summary": summary,
        "write_stats": st,
        "common_recall_anchors": anchors,
        "comparison": vals,
        "limitations": [
            "All repetitions use same frozen query stream; not independent user samples",
            "SSD reads are logical; physical write amplification and endurance not measured",
            "Sample sampling rate randomized; online state updates after completed queries",
            "Reads are matched-recall; L160 write rates are fixed-L, not the same Pareto point",
        ],
    }
    save(args.out / "final-core-write-sweep.json", results)
    print("FINAL_CORE_PUBMED_WRITE_SWEEP_PASS", json.dumps(vals, indent=2), flush=True)


if __name__ == "__main__":
    main()
