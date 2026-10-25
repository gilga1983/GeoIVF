#!/usr/bin/env python3
"""Reproduce the paper's RAM-sensitivity CSVs from two successful GeoIVF ZIPs.

Full Sample2 1x--16x: run 37886564510, navhints-memory-sweep artifact.
Joint RAM-only Core controls: run 37886413522, competitor-memory-sweep artifact.
Compare absolute SSD reads at matching recalls BEFORE averaging percentage
savings; never pool runs or extrapolate beyond a measured recall frontier.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from zipfile import ZipFile

FULL_MEMBER = "results/navhints-memory-sweep.json"
CORE_MEMBER = "results/competitor-memory-sweep.json"
FULL_FIELDS = (
    "method", "tier", "auxiliary_payload_bytes", "recall_anchors",
    "full_read_saving_percent", "full_latency_saving_percent",
)
CORE_FIELDS = (
    "method", "auxiliary_payload_bytes", "rep_anchor_observations",
    "core_read_saving_percent", "full_read_saving_percent",
)


def load_archive(path: Path, member: str) -> dict:
    with ZipFile(path) as archive:
        return json.loads(archive.read(member))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def full_rows(result: dict) -> list[dict]:
    protocol = result["protocol"]
    assert protocol["repetitions"] == 3
    assert protocol["controller_query_side_payload_bytes"] == 102940
    assert (protocol["K"], protocol["beam"], protocol["threads"]) == (10, 8, 4)
    rows = []
    for method, comparison in result["matched_at_navhints"].items():
        if method in {"baseline", "navhints-full"}:
            continue
        options = result["memory"][method]
        available = [a for a in comparison["per_anchor"] if a["available"]]
        assert len(available) == comparison["common_anchor_count"], method
        io = statistics.mean(a["navhints_io_saving_percent"] for a in available)
        latency = statistics.mean(a["navhints_latency_saving_percent"] for a in available)
        assert abs(io - comparison["mean_navhints_io_saving_percent"]) < 1e-7
        assert abs(latency - comparison["mean_navhints_latency_saving_percent"]) < 1e-7
        rows.append({
            "method": method, "tier": options.get("tier", 1),
            "auxiliary_payload_bytes": options["aux_bytes_estimated"],
            "recall_anchors": len(available),
            "full_read_saving_percent": f"{io:.5f}",
            "full_latency_saving_percent": f"{latency:.5f}",
        })
    return sorted(rows, key=lambda x: (int(x["tier"]), x["method"]))


def core_rows(result: dict) -> list[dict]:
    assert result["protocol"]["repetitions"] == 3
    memory = result["memory"]
    assert memory["core_query_payload_bytes"] == 102940
    comps = result["matched_comparisons"]
    controls = []
    core_by_rep = {}
    for row in comps["core"]["matched_anchors"]:
        core_by_rep.setdefault(row["rep"], {})[row["anchor_L"]] = row
    for method in ("hot-30", "hot-512", "cat-80", "cat-640"):
        vals, full_vals = [], []
        for comparison in comps[method]["matched_anchors"]:
            core = core_by_rep[comparison["rep"]].get(comparison["anchor_L"])
            if core is None:
                continue
            assert abs(core["reference_reads"] - comparison["reference_reads"]) < 1e-5
            vals.append(100 * (1 - core["comparison_reads"] / comparison["comparison_reads"]))
            full_vals.append(comparison["sample2_saving_percent"])
        assert len(vals) >= 12, method
        if method.startswith("hot-"):
            aux = memory["hot_node_memory_upper_estimate_bytes"][method.removeprefix("hot-")]
        else:
            aux = memory["catapult_reserved_bytes"][method.removeprefix("cat-")]
        controls.append({
            "method": method, "auxiliary_payload_bytes": aux,
            "rep_anchor_observations": len(vals),
            "core_read_saving_percent": f"{statistics.mean(vals):.5f}",
            "full_read_saving_percent": f"{statistics.mean(full_vals):.5f}",
        })
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide-artifact", type=Path, required=True)
    parser.add_argument("--core-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/data"))
    args = parser.parse_args()
    wide = full_rows(load_archive(args.wide_artifact, FULL_MEMBER))
    core = core_rows(load_archive(args.core_artifact, CORE_MEMBER))
    write_csv(args.output_dir / "eval_memory_scaling.csv", FULL_FIELDS, wide)
    write_csv(args.output_dir / "eval_memory_scaling_core.csv", CORE_FIELDS, core)
    print(f"Verified {len(wide)} RAM sensitivity configurations and {len(core)} Core controls")
    for row in wide:
        if row["tier"] == 16:
            print(f"16x {row['method']}: {row['full_read_saving_percent']}% fewer SSD reads")
    for row in core:
        if row["method"] in ("hot-512", "cat-640"):
            print(f"Core vs {row['method']}: {row['core_read_saving_percent']}% fewer SSD reads")


if __name__ == "__main__":
    main()
