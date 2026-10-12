#!/usr/bin/env python3
"""Audit native Starling component pruning and compact-budget comparisons.

Only original-native Starling runs on the same frozen index/query stream are
compared. Interpolate three-repetition mean ABSOLUTE recall-cost curves at
common recall targets. Missing jointly supported anchors remain unavailable.
Ablations are exploratory; choose a champion using separate validation data,
not by optimizing this held-out report.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

MODES = {
    "components": (
        "baseline", "final_core", "junction_only", "destination_only"
    ),
    "compact": (
        "baseline", "final_core", "junction_only", "destination_only",
        "junction_compact", "destination_compact", "joint_compact"
    ),
}
BUDGETS = {
    "baseline": (0, 0, 0),
    "final_core": (14464, 2048, 8),
    "junction_only": (14464, 0, 0),
    "destination_only": (0, 2048, 8),
    "junction_compact": (4096, 0, 0),
    "destination_compact": (0, 512, 2),
    "joint_compact": (4096, 512, 2),
}
DATASETS = {
    "coveo": ([12,20,40,80], 5000, 20000, [97.,98.]),
    "bigann": ([20,40,80,160], 1000, 4000, [97.,98.]),
    "pubmed": ([20,40,80,160,320,640], 1000, 4000, [99.,99.5]),
    "text2image": ([20,40,80,160,320,640], 1000, 4000, [60.,80.]),
}
REPS = (0, 1, 2)
MEASURES = ("ios","total_us","cpu_us","cache_hits")


def recover_dataset(path: Path) -> str:
    matches = [d for d in DATASETS if f"starling-prune-budget-{d}-" in path.name]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify a unique native dataset in {path.name}")
    return matches[0]


def load(root: Path, modes: tuple[str, ...], dataset: str) -> dict:
    widths, expected, warmup, _ = DATASETS[dataset]
    records = {}
    for rep in REPS:
        for mode in modes:
            filename = root / f"starling-rep{rep}-{mode}-summary.txt"
            lines = filename.read_text().splitlines()
            for L in widths:
                row = [line.split() for line in lines
                       if line.split() and line.split()[0] == str(L)]
                if len(row) != 1 or len(row[0]) < 10:
                    raise ValueError(f"Missing native recall row: {filename} L={L}")
                recall = float(row[0][-1])
                if not 0 <= recall <= 100:
                    raise ValueError(f"Invalid native recall: {filename} L={L}")
                csv_path = root / "diagnostics" / f"rep{rep}-{mode}" / f"L{L}.csv"
                with csv_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != expected:
                    raise ValueError(f"Wrong query count: {csv_path}")
                if [int(x["query"]) for x in rows] != list(range(expected)):
                    raise ValueError(f"Nonconsecutive query sequence: {csv_path}")
                means = {}
                for field in MEASURES:
                    values = [float(x[field]) for x in rows]
                    if any(value < 0 for value in values):
                        raise ValueError(f"Negative {field}: {csv_path}")
                    means[field] = statistics.mean(values)
                records[(mode,L,rep)] = dict(means, recall=recall)
    assert len(records) == len(modes)*len(widths)*len(REPS)
    return {
        "widths": widths, "measured_queries": expected,
        "warmup_queries": warmup, "records": records
    }


def interpolate(rows: dict[int,dict], target: float):
    # Interpolate absolute logical SSD read and latency values, with no
    # extrapolation or substitution of results from a different dataset.
    ordered = sorted((x["recall"], x["ios"], x["total_us"], x["cpu_us"])
                     for x in rows.values())
    if target < ordered[0][0]-1e-7 or target > ordered[-1][0]+1e-7:
        return None
    for p in ordered:
        if abs(p[0]-target) < 1e-7:
            return {"ios":p[1], "total_us":p[2], "cpu_us":p[3]}
    for first, second in zip(ordered, ordered[1:]):
        if first[0] < target < second[0]:
            f = (target-first[0])/(second[0]-first[0])
            return dict(zip(("ios","total_us","cpu_us"),
                   [v+f*(w-v) for v,w in zip(first[1:],second[1:])]))
    return None


def report(root: Path, stage: str):
    dataset = recover_dataset(root)
    modes = MODES[stage]
    evidence = load(root, modes, dataset)
    widths = evidence["widths"]
    per_L = {}
    for mode in modes:
        per_L[mode] = {}
        for L in widths:
            values = [evidence["records"][(mode,L,rep)] for rep in REPS]
            per_L[mode][L] = {
                key:statistics.mean(x[key] for x in values)
                for key in (*MEASURES,"recall")
            }

    comparison = {}
    targets = DATASETS[dataset][3]
    for target in targets:
        baseline = interpolate(per_L["baseline"], target)
        core = interpolate(per_L["final_core"], target)
        arr = {}
        for mode in modes:
            value = interpolate(per_L[mode], target)
            if value is None or baseline is None or core is None:
                arr[mode] = None
            else:
                arr[mode] = {
                    "reads_per_query":value["ios"],
                    "search_latency_us":value["total_us"],
                    "cpu_time_us":value["cpu_us"],
                    "read_savings_vs_native_pct":
                        100*(1-value["ios"]/baseline["ios"]),
                    "latency_savings_vs_native_pct":
                        100*(1-value["total_us"]/baseline["total_us"]),
                    "read_savings_vs_full_navhints_pct":
                        100*(1-value["ios"]/core["ios"]),
                    "latency_savings_vs_full_navhints_pct":
                        100*(1-value["total_us"]/core["total_us"]),
                }
        comparison[str(target)] = arr
    return {
        "dataset":dataset,"stage":stage,
        "native_source":"zilliztech/starling@17dc3e8a011533a62374445f53963e951b72883a",
        "modes":[{"name":m,"B":BUDGETS[m][0],"C":BUDGETS[m][1],
                  "k":BUDGETS[m][2]} for m in modes],
        "protocol":{"repetitions":3,"widths":widths,
                    "static_training_queries":5000,
                    "causal_warmup_queries":evidence["warmup_queries"],
                    "measured_queries":evidence["measured_queries"],
                    "same_frozen_native_index":True,
                    "unchanged_native_navigator":True,
                    "result_selection_on_heldout_data":False},
        "per_width":{m:{str(L):r for L,r in obj.items()}
                     for m,obj in per_L.items()},
        "matched_recall":comparison,
        "warnings":[
            "Native provider reads are logical SSD operations.",
            "Search timing may exclude post-query history updates.",
            "Three repetitions share an identical frozen query sequence.",
            "This is a controlled ablation, not a selected champion."
        ],
    }


def markdown(data: dict) -> str:
    lines = [
        f"# Original Starling component pruning: {data['dataset']}",
        "",
        f"Stage: {data['stage']}. Same native Starling graph and navigator.",
        "Positive savings denote improvements; negative values denote extra cost.",
        "",
        "| Mode | B | C | Starts |",
        "|---|---:|---:|---:|",
    ]
    for mode in data["modes"]:
        lines.append(f"| {mode['name']} | {mode['B']} | {mode['C']} | {mode['k']} |")
    for target, values in data["matched_recall"].items():
        lines += [
            "",f"## Recall@10: {target}%", "",
            "| Mode | Reads/q | Search us | Reads saved vs native | Latency saved vs native |",
            "|---|---:|---:|---:|---:|"
        ]
        for mode in data["modes"]:
            name = mode["name"]
            row = values[name]
            if row is None:
                lines.append(f"| {name} | unavailable | unavailable | unavailable | unavailable |")
            else:
                lines.append(
                    f"| {name} | {row['reads_per_query']:.3f} | "
                    f"{row['search_latency_us']:.2f} | "
                    f"{row['read_savings_vs_native_pct']:+.2f}% | "
                    f"{row['latency_savings_vs_native_pct']:+.2f}% |")
    return "\n".join(lines)+"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics",type=Path)
    parser.add_argument("output_json",type=Path)
    parser.add_argument("output_md",type=Path)
    parser.add_argument("--stage",choices=tuple(MODES),required=True)
    args = parser.parse_args()
    root = args.diagnostics.resolve().parent
    result = report(root,args.stage)
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    args.output_md.write_text(markdown(result))
    print("STARLING_PRUNE_BUDGET_REPORT_VERIFIED",result["dataset"],
          result["stage"],"native=1",flush=True)
    print(markdown(result),flush=True)


if __name__ == "__main__":
    main()
