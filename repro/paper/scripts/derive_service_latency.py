#!/usr/bin/env python3
"""Regenerate NavHints complete-service figure CSVs and the latency table.

Inputs: four successful GeoIVF Actions artifact ZIPs, with exactly one
results/complete-service-*.json in each archive. Every retained recall target
must be supported by all repetitions and all three Catapult seeds. Remove
per-run dominated recall/mean-latency points before interpolating absolute
request time. The two original papers' other mechanism ablations are unchanged.

Example:
 python3 scripts/derive_service_latency.py \
   --coveo coveo.zip --pubmed pubmed.zip --text2image t2i.zip \
   --bigann bigann.zip --out-data figures/data \
   --out-table tables/eval_service_latency_table.tex \
   --out-summary /tmp/eval_service_latency_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

DATASETS = (
    ("coveo", "coveo", "eval_service_coveo.csv", "Coveo"),
    ("pubmed", "pubmed", "eval_service_pubmed.csv", "PubMed"),
    ("text2image", "text2image-10M", "eval_service_t2i.csv", "T2I-10M"),
    ("bigann", "bigann-10M", "eval_service_bigann.csv", "BigANN-10M"),
)


def read_zip(path):
    with zipfile.ZipFile(path) as archive:
        reports = [n for n in archive.namelist()
                   if n.startswith("results/complete-service-") and n.endswith(".json")]
        if len(reports) != 1:
            raise ValueError(f"{path}: expected one complete-service result")
        return json.loads(archive.read(reports[0]))


def pareto(points):
    """Retain only measured nondominated recall/mean-latency operating points."""
    if not points or any(not (0 <= p["recall"] <= 100 and p["mean_service_us"] > 0)
                         for p in points):
        raise ValueError("invalid recall/latency observations")
    rows = sorted(points, key=lambda x: (x["recall"], x["mean_service_us"]))
    keep = [a for a in rows if not any(
        b["recall"] >= a["recall"] - 1e-8
        and b["mean_service_us"] <= a["mean_service_us"] + 1e-8
        and (b["recall"] > a["recall"] + 1e-8
             or b["mean_service_us"] < a["mean_service_us"] - 1e-8)
        for b in rows)]
    if not keep:
        raise ValueError("empty Pareto frontier")
    return keep


def interpolate(points, recall):
    if recall < points[0]["recall"] - 1e-8 or recall > points[-1]["recall"] + 1e-8:
        return None
    for point in points:
        if abs(recall-point["recall"]) < 1e-8:
            return point["mean_service_us"]
    for left, right in zip(points, points[1:]):
        if left["recall"] < recall < right["recall"]:
            weight = (recall-left["recall"]) / (right["recall"]-left["recall"])
            return left["mean_service_us"] + weight * (
                right["mean_service_us"]-left["mean_service_us"])
    return None


def calculate(report, dataset):
    if dataset == "pubmed":
        widths = report["widths"]
        repetitions = report["repetitions"]
        seeds = report["catapult_seeds"]
        B = report["controller"]["junctions"]
        C, k = report["controller"]["recent"], report["controller"]["starts"]
    else:
        if report["dataset"] != dataset:
            raise ValueError("wrong workload in native artifact")
        widths = report["protocol"]["widths"]
        repetitions = report["protocol"]["rotated_repetitions"]
        seeds = report["catapult"]["seeds"]
        B = report["core"]["actual_junctions"]
        C, k = report["core"]["recent"], report["core"]["starts"]
        if report["protocol"]["in_flight_queries"] != 1:
            raise ValueError("not a strict causal replay")
        if report["catapult"]["accounted_payload_bytes"] < report["core"]["accounted_payload_bytes"]:
            raise ValueError("Catapult routing payload undersized")
    if (report.get("diagnostic") or repetitions != 3 or sorted(seeds) != [0, 1, 2]
            or not (0 < B <= 14464 and C == 2048 and k == 8)):
        raise ValueError("not the selected three-repetition final Core experiment")

    measurements = report["measurements"]

    def curve(rep, arm):
        return pareto([{"L": L, **measurements[f"r{rep}-L{L}-{arm}"]}
                       for L in widths])

    matched = []
    for L in widths:
        target = statistics.mean(
            measurements[f"r{rep}-L{L}-baseline"]["recall"]
            for rep in range(repetitions))
        costs = {}
        for arm in ("baseline", "navhints", "catapult"):
            values = []
            for rep in range(repetitions):
                for seed in (seeds if arm == "catapult" else [None]):
                    method = f"catapult-s{seed}" if arm == "catapult" else arm
                    value = interpolate(curve(rep, method), target)
                    if value is not None:
                        values.append(value)
            if len(values) != repetitions * (len(seeds) if arm == "catapult" else 1):
                break
            costs[arm] = statistics.mean(values)
        if len(costs) == 3:
            matched.append({
                "recall": target, "L": L,
                "save_diskann": 100 * (1-costs["navhints"]/costs["baseline"]),
                "save_catapult": 100 * (1-costs["navhints"]/costs["catapult"]),
            })
    if len(matched) < 2:
        raise ValueError(f"{dataset}: too few common efficient-frontier targets")

    plot = {}
    for arm in ("baseline", "catapult", "navhints"):
        rows = []
        for L in widths:
            samples = [measurements[
                f'r{rep}-L{L}-{arm if arm != "catapult" else "catapult-s"+str(seed)}'
            ] for rep in range(repetitions)
               for seed in (seeds if arm == "catapult" else [None])]
            rows.append({
                "L": L,
                "recall": statistics.mean(s["recall"] for s in samples),
                "mean_service_us": statistics.mean(s["mean_service_us"] for s in samples),
            })
        plot[arm] = {p["L"]: p for p in pareto(rows)}
    return {
        "targets": len(matched), "min_recall": min(p["recall"] for p in matched),
        "max_recall": max(p["recall"] for p in matched),
        "save_diskann": statistics.mean(p["save_diskann"] for p in matched),
        "save_catapult": statistics.mean(p["save_catapult"] for p in matched),
        "matched": matched, "plot": plot, "widths": widths,
    }


def write_csv(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        csvwriter = csv.writer(stream)
        csvwriter.writerow(["L", "baseline_recall", "baseline_ms",
                            "catapult_recall", "catapult_ms",
                            "navhints_recall", "navhints_ms"])
        for L in report["widths"]:
            row = [L]
            for arm in ("baseline", "catapult", "navhints"):
                value = report["plot"][arm].get(L)
                row.extend((f'{value["recall"]:.6f}',
                            f'{value["mean_service_us"]/1000:.6f}')
                           if value else ("nan", "nan"))
            csvwriter.writerow(row)


def latex_table(rows):
    output = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean complete-request latency savings of \sys relative to unmodified \DiskANN and source-aligned \CatapultDB routing at matched Recall@10. The range and number of jointly supported targets are workload specific. Positive values mean faster requests; the per-target mean savings are averaged after removing dominated operating points within each repetition and Catapult seed.}",
        r"\label{tab:eval-service-latency}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Workload & Recall@10 ($n$) & DiskANN & Catapult \\",
        r"\midrule",
    ]
    for item in rows:
        d, x = item["label"], item["result"]
        output.append(
            f'{d} & {x["min_recall"]:.2f}--{x["max_recall"]:.2f}\\% ({x["targets"]}) & '
            f'{x["save_diskann"]:+.1f}\\% & {x["save_catapult"]:+.1f}\\% ' + r"\\"
        )
    output += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("coveo", "pubmed", "text2image", "bigann"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out-data", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for key, dataset, filename, label in DATASETS:
        path = getattr(args, key)
        result = calculate(read_zip(path), dataset)
        write_csv(args.out_data / filename, result)
        result.pop("plot")
        rows.append({
            "label": label, "result": result,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_table.write_text(latex_table(rows))
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        r = row["result"]
        print(row["label"], r["targets"],
              f'vs DiskANN {r["save_diskann"]:.2f}%, '
              f'vs Catapult {r["save_catapult"]:.2f}%')


if __name__ == "__main__":
    main()
