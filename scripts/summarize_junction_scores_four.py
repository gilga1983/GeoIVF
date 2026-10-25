#!/usr/bin/env python3
"""Validate and tabulate four-workload NavHints junction-score ablation.

Fail closed: all four successful real-SSD workflow artifacts, original
directory reproduction, selected memory configuration, 3 repetitions and
joint matched-recall anchors are required. Unfinished or old failed runs
must never produce a paper table.

Inputs: ZIPs from PubMed, BigANN, Coveo B8192, Text2Image run artifacts.
Output: JSON evidence + a LaTeX table suitable for RQ2 only after review.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import zipfile
from pathlib import Path

META = {
    "coveo": {
        "name": "Coveo",
        "expected_b": 8192,
        "status": "NAVHINTS_SCORE_coveo_PASS",
        "repro": "ORIGINAL_FULL_COVEO_DIRECTORY_BYTE_IDENTICAL",
    },
    "pubmed": {
        "name": "PubMed",
        "expected_b": 14464,
        "status": "NAVHINTS_JUNCTION_SCORE_pubmed_PASS",
        "repro": "SELECTED_NAVHINTS_ORIGINAL_REPRODUCED",
    },
    "text2image": {
        "name": "Text2Image-10M",
        "expected_b": 14464,
        "status": "NAVHINTS_SCORE_text2image_PASS",
        "repro": "ORIGINAL_T2I_DIRECTORY_BYTE_IDENTICAL",
    },
    "bigann": {
        "name": "BigANN-10M",
        "expected_b": 14464,
        "status": "NAVHINTS_JUNCTION_SCORE_bigann_PASS",
        "repro": "SELECTED_NAVHINTS_ORIGINAL_REPRODUCED",
    },
}


def read(archive_path, key):
    expected = META[key]
    with zipfile.ZipFile(archive_path) as z:
        names = set(z.namelist())
        for name in (
            "results/junction-score-ablation.json",
            "selection/selection.json",
            "repro.status",
            "status.txt",
        ):
            if name not in names:
                raise ValueError(f"{key}: evidence missing {name}")
        if z.read("status.txt").decode().strip() != expected["status"]:
            raise ValueError(f"{key}: missing successful final run marker")
        if z.read("repro.status").decode().strip() != expected["repro"]:
            raise ValueError(f"{key}: frozen directory reference not reproduced")
        report = json.loads(z.read("results/junction-score-ablation.json"))
        selection = json.loads(z.read("selection/selection.json"))
    if selection["budget"] != expected["expected_b"] or selection["queries"] != 5000:
        raise ValueError(f"{key}: selection budget or teacher prefix differs")
    if not selection.get("ranking_prefix_reproduced", True):
        raise ValueError(f"{key}: frozen sum-scoring ID reference unverified")
    cfg = report["controller"]
    if (
        cfg.get("B") != expected["expected_b"]
        or cfg.get("C") != 2048
        or cfg.get("k") != 8
        or cfg.get("regions") != 512
        or cfg.get("persistent_exits") != 0
    ):
        raise ValueError(f"{key}: scoring study changed selected Core")
    if key == "coveo" and cfg.get("full_selected_B") != 12169:
        raise ValueError("Coveo reduced-budget sensitivity not clearly marked")
    if report["protocol"]["repetitions"] != 3:
        raise ValueError(f"{key}: not three rotated repeated replays")
    if set(report["savings_vs_alternative"]) != {"count", "mean"}:
        raise ValueError("Expected both comparison modes")
    matched = report["matched"]
    if len(matched) < 3:
        raise ValueError(f"{key}: fewer than 3 common recall targets")
    seen = []
    for row in matched:
        recall = float(row["recall_percent"])
        if not math.isfinite(recall) or not 0 <= recall <= 100:
            raise ValueError("Invalid recall")
        # Distinct search widths can yield identical recall targets:
        # do not weight a repeated recall as separate evidence.
        if any(abs(recall - q) < 1e-8 for q in seen):
            raise ValueError(f"{key}: repeated recall target detected")
        seen.append(recall)
        costs = row["reads"]
        if set(costs) != {"sum", "count", "mean"} or any(
            not math.isfinite(x) or x <= 0 for x in costs.values()
        ):
            raise ValueError("Invalid per-target SSD reads")
        for other in ("count", "mean"):
            expected_delta = 100 * (1 - costs["sum"] / costs[other])
            if not math.isclose(
                row[f"sum_vs_{other}_saving_percent"], expected_delta, abs_tol=1e-8
            ):
                raise ValueError(f"{key}: mismatched score differential")
    result = {
        "dataset": expected["name"],
        "B": cfg["B"],
        "targets": len(matched),
        "min_recall": min(seen),
        "max_recall": max(seen),
    }
    for mode in ("count", "mean"):
        vals = [float(x[f"sum_vs_{mode}_saving_percent"]) for x in matched]
        avg = statistics.mean(vals)
        archive = report["savings_vs_alternative"][mode]
        if archive["targets"] != len(vals) or not math.isclose(
            avg, archive["mean_read_saving_pct"], abs_tol=1e-8
        ):
            raise ValueError(f"{key}: inconsistent archived mean")
        result[f"vs_{mode}_pct"] = avg
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in META:
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--tex", type=Path, required=True)
    a = p.parse_args()
    order = ("coveo", "pubmed", "text2image", "bigann")
    results = {key: read(getattr(a, key), key) for key in order}
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.tex.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(
        json.dumps(
            {"source": "frozen, verified real-SSD artifacts", "results": results},
            indent=2,
        )
        + "\n"
    )
    parts = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Junction-ranking sensitivity.} Mean matched-recall "
        r"logical SSD-read savings of the selected sum-of-expansion-positions "
        r"score over two alternative rankings, using the same routing "
        r"budget and query sequence within each workload. Positive values "
        r"favor the selected score. Coveo uses $B=8{,}192$ to make selection "
        r"binding (the reference design retains all 12,169 eligible IDs); "
        r"other workloads use $B=14{,}464$.}",
        r"\label{tab:eval-junction-scoring}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Workload & Targets & vs.\ frequency & vs.\ mean position \\",
        r"\midrule",
    ]
    for key in order:
        row = results[key]
        parts.append(
            f"{row['dataset']} & {row['targets']} & "
            f"{row['vs_count_pct']:+.2f}" + r"\% & "
            f"{row['vs_mean_pct']:+.2f}" + r"\% \\"
        )
    parts += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    a.tex.write_text("\n".join(parts) + "\n")
    print("NAVHINTS_FOUR_WORKLOAD_JUNCTION_SCORE_VERIFIED")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
