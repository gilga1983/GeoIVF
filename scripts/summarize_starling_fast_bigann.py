#!/usr/bin/env python3
"""Strict original-Starling BigANN10M classic vs PQ-optimized Core validation."""
import json
from pathlib import Path
import sys
import summarize_starling_fast_core as fast


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: analyzer DIAGNOSTICS OUT_JSON OUT_MD")
    fast.LS=(20,40,80,160)
    fast.QUERY_COUNT=1000
    fast.SOURCE_EXPECTED={
        (r,m,l) for r in fast.REPS for m in fast.MODES for l in fast.LS}
    artifact=Path(sys.argv[1]).resolve().parent
    result=fast.analyze(artifact)
    result["metadata"].update({
        "dataset":"BigANN-10M public benchmark, 10,000,000 × 128 uint8, squared L2",
        "protocol":"pinned original Starling; same native graph for all arms; "
                   "5K disjoint training queries; 4K causal warmup, "
                   "last 1K measured; three rotated repetitions, SSD lock",
        "comparison":"same-sized 16K Starling-native learned junction ID directory "
                     "and 512 causal recent successful IDs",
        "dataset_locality":"weak recent destination overlap compared with chronological Coveo",
        "validated_pair_rows_expected":24000,
        "claim_boundaries":"native original Starling logical I/O counters, "
                          "native per-query µs timing; not physical NVMe blocks. "
                          "No cross-system speed ratios; same-L classic/fast recall identical. "
                          "Starling+Core vs native Starling may differ in recall."
    })
    if result["validated_route_pairs"]!=24000:
        raise RuntimeError(f"expected 24K exact pair matches, got {result['validated_route_pairs']}")
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
    report=fast.markdown(result)
    report=report.replace(
        "# Starling native Core packed-PQ speedup: Coveo causal workload",
        "# Original Starling on BigANN-10M: exact-decision PQ speedup")
    report=report.replace(
        "Three self-hosted repetitions, original native Starling index, 31,950 × 50 normalized product vectors; 5K static / 20K causal warm / 5K eval.",
        "Three self-hosted repetitions on frozen BigANN-10M 128D uint8/L2, "
        "original native Starling graph and memory navigator; "
        "5K disjoint static / 4K causal warm / 1K eval.")
    Path(sys.argv[3]).write_text(report)
    print("STARLING_FAST_BIGANN_24K_PAIRED_ROUTES_VERIFIED",flush=True)
    print(report,flush=True)


if __name__=="__main__":main()
