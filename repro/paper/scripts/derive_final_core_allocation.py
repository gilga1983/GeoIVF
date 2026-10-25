#!/usr/bin/env python3
"""Audit selected-Core PubMed fixed-payload allocation against Core itself.

Input: successful GeoIVF 38120556463 artifact 11696380166, containing
results/navhints-memory-tradeoff.json. The older reference in that
artifact is only a source of raw experimental arms, not a paper method.
Outputs the selected-Core-anchored eight-point comparison used in
Table eval-core-reallocation; no extrapolation or averaging of ratios.
"""
import argparse
import json
import statistics
import zipfile
from pathlib import Path

SELECTED = "equal-14464-2048-k8"
CONTROL_ARMS = (
    "equal-14464-2048-k4",
    "equal-14464-2048-k2",
    "equal-15488-1024-k4",
    "equal-12416-4096-k4",
    "equal-8320-8192-k4",
)

def curve(raw):
    vals = sorted((float(v["recall"]), float(v["reads"])) for v in raw.values())
    if len(vals) != 9 or any(a[0] >= b[0] for a, b in zip(vals, vals[1:])):
        raise ValueError("Expected nine strictly increasing recall points")
    return vals

def at(values, recall):
    if recall < values[0][0] - 1e-7 or recall > values[-1][0] + 1e-7:
        return None
    for x, y in values:
        if abs(x - recall) < 1e-7:
            return y
    for (x, y), (x2, y2) in zip(values, values[1:]):
        if x < recall < x2:
            return y + (recall - x) * (y2 - y) / (x2 - x)
    return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    with zipfile.ZipFile(args.artifact) as z:
        raw = json.loads(z.read("results/navhints-memory-tradeoff.json"))
    if raw["accounted_byte_budget"] != 102940:
        raise ValueError("Mismatched accounted Core payload")
    results = raw["results"]
    curves = {k: curve(results[k]) for k in (SELECTED, *CONTROL_ARMS)}
    selected_targets = [r for r, _ in curves[SELECTED]]
    common = [
        r for r in selected_targets
        if all(at(curves[k], r) is not None for k in curves)
    ]
    if len(common) != 8:
        raise ValueError(f"Expected eight jointly reachable selected-Core recalls, got {common}")
    if not 38.85 < min(common) < 38.87 or not 78.84 < max(common) < 78.86:
        raise ValueError("Unexpected matched Core recall range")
    summary = {}
    for arm in (SELECTED, *CONTROL_ARMS):
        extras = []
        for r in common:
            core = at(curves[SELECTED], r)
            competitor = at(curves[arm], r)
            extras.append(100 * (competitor / core - 1))
        summary[arm] = {
            "extra_reads_percent_vs_selected_core": statistics.mean(extras),
            "per_target_extra_reads_percent": extras,
        }
    output = {
        "source": "GeoIVF run 38120556463 artifact 11696380166",
        "selected_core": {"B": 14464, "C": 2048, "k": 8, "regions": 512},
        "accounted_payload_bytes": 102940,
        "matched_core_recalls": common,
        "arms": summary,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
