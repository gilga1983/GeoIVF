#!/usr/bin/env python3
"""Qualify first-time native Starling selected Core on PubMed and T2I-10M.

Exactly original author native Starling + preserved navigator/page layout,
with the selected final Core B14464/C2048/eight-ranked-start extension.
Averaging/interpolation follows the existing Table 9 convention.
Do not extrapolate missing 97/98 Recall@10 measurements.
"""
from __future__ import annotations
from pathlib import Path
import json, sys
from summarize_starling_final_core import analyze, markdown
import summarize_starling_final_core as native

LS = (20, 40, 80, 160, 320, 640)
REPORT_TARGETS = (97.0, 98.0)


def interpolated(points, target, field):
    arr = sorted((float(v["recall"]), float(v[field])) for v in points.values())
    if len(arr) != len(LS) or target < arr[0][0] or target > arr[-1][0]:
        return None
    for i, (r, x) in enumerate(arr):
        if abs(r - target) < 1e-9:
            return x
        if i and arr[i - 1][0] < target < r:
            r0, x0 = arr[i - 1]
            return x0 + (target - r0) * (x - x0) / (r - r0)
    return None


def report(data):
    baseline = data["intra_native_methods"]["baseline"]
    core = data["intra_native_methods"]["final_core"]
    support = [
        max(min(float(v["recall"]) for v in a.values()) for a in (baseline, core)),
        min(max(float(v["recall"]) for v in a.values()) for a in (baseline, core)),
    ]
    table = {}
    for t in REPORT_TARGETS:
        a = interpolated(baseline, t, "ios")
        b = interpolated(core, t, "ios")
        c = interpolated(baseline, t, "total_us")
        d = interpolated(core, t, "total_us")
        table[str(t)] = (
            None
            if None in (a, b, c, d)
            else {
                "target_recall": t,
                "native_baseline_reads": a,
                "native_core_reads": b,
                "native_baseline_latency_us": c,
                "native_core_latency_us": d,
                "read_saving_percent": 100 * (1 - b / a),
                "latency_saving_percent": 100 * (1 - d / c),
            }
        )
    return {
        "supported_recall_range": support,
        "table9_fixed_targets": table,
        "all_fixed_targets_reachable": all(x is not None for x in table.values()),
        "reason_if_missing": "97%/98% target outside common native recall support; do not extrapolate.",
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: summarize_starling_final_core_missing.py DIAG_FOLDER OUTPUT_JSON OUTPUT_MD"
        )
    diag = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    md = Path(sys.argv[3])
    root = diag.parent
    if "starling-final-core-pubmed" in str(root):
        ds = "PubMed1M/MedCPT"
        name = "pubmed"
    elif "starling-final-core-text2image" in str(root):
        ds = "Text2Image-10M"
        name = "text2image"
    else:
        raise ValueError("No original Starling Table 9 evidence for " + str(root))
    native.path_config = lambda _: (LS, 1000, ds, 4000)
    data = analyze(root)
    data["native_distance"] = "mips (maximum inner product), float32"
    data["original_starling_source"] = "17dc3e8a011533a62374445f53963e951b72883a"
    data["new_dataset_only"] = name
    data["fixed_table9"] = report(data)
    data["input_verification"] = (
        "5K disjoint training; 4K causal online warm-up; 1K measured"
    )
    if data["protocol"]["repetitions"] != 3 or data["protocol"]["L"] != list(LS):
        raise AssertionError("Changed native benchmark protocol")
    if data["measured_replay_count"] != len(LS) * 3 * 3:
        raise AssertionError("Expected all 3 native arms x 3 reps x 6 widths")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    md.write_text(
        markdown(data)
        + "\n## Table 9 fixed recall targets\n\n"
        + json.dumps(data["fixed_table9"], indent=2)
        + "\n"
    )
    print(
        "STARLING_TABLE9_NEW_NATIVE_QUALIFIED",
        name,
        json.dumps(data["fixed_table9"], sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
