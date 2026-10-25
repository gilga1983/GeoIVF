#!/usr/bin/env python3
"""Offline verification of the data-minimized NavHints PVLDB reviewer bundle.

No dependencies beyond Python 3.10's standard library. Does not fetch datasets,
compile an ANN engine, execute an SSD benchmark, or modify the paper.

  python3 repro/verify_review_bundle.py                 # source/provenance audit
  python3 repro/verify_review_bundle.py --bundle path.zip

The public aggregate ZIP is bundled at repro/navhints-reviewer-evidence.zip.
Original input datasets and raw experiment archives are not distributed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("repro/manifest.json", 14, "original_artifact_sha256"),
    ("repro/supplemental-manifest-20261014.json", 9, "sha256"),
    ("repro/active-figures-table7-sources-20261017.json", 7, "sha256"),
    ("repro/destination-controls-sources-20261025.json", 2, "sha256"),
]
DATASETS = {
    "coveo": ("Coveo", "coveo-final-core.json"),
    "pubmed": ("PubMed1M/MedCPT", "public-final-core.json"),
    "text2image": ("Text2Image-10M", "public-final-core.json"),
    "bigann": ("BigANN-10M", "public-final-core.json"),
}
SERVICE = {
    "coveo": ("complete-service-coveo.json", 3),
    "pubmed": ("complete-service-pubmed.json", 10),
    "text2image": ("complete-service-text2image-10M.json", 4),
    "bigann": ("complete-service-bigann-10M.json", 4),
}
STARTS = (1, 2, 4, 8)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def is_hash(value: object, size: int) -> bool:
    return isinstance(value, str) and len(value) == size and all(c in "0123456789abcdef" for c in value)


def check_source_manifests() -> dict[str, str]:
    """Public source inventory and cited result contracts; no data download."""
    digest_to_id = {}
    for filename, count, digest_key in SOURCES:
        record = json.loads((ROOT / filename).read_text())
        runs = record["runs"]
        require(len(runs) == count, f"{filename}: expected {count} sources, observed {len(runs)}")
        for row in runs:
            digest = row[digest_key]
            require(is_hash(digest, 64), f"{filename}: invalid checksum {row['id']}")
            require(digest not in digest_to_id, f"duplicate source ZIP hash: {digest}")
            require(isinstance(row.get("run_id"), int) and isinstance(row.get("artifact_id"), int),
                    f"{filename}: missing original Actions provenance for {row['id']}")
            digest_to_id[digest] = row["id"]
    summary = json.loads((ROOT / "repro/current-paper-summary-20261017.json").read_text())
    require(len(summary["headline_results"]) == 4, "public summary does not cover all workloads")
    require(len(summary["junction_scoring"]["workloads"]) == 4, "missing scoring workload")
    return digest_to_id


def archive_checks(z: zipfile.ZipFile, public_digests: dict[str, str]) -> dict:
    files = z.infolist()
    names = [f.filename for f in files]
    require(len(names) == len(set(names)), "duplicate ZIP entries")
    for f in files:
        path = Path(f.filename)
        require(not path.is_absolute() and ".." not in path.parts and not f.filename.startswith("/"),
                "unsafe ZIP member name")
        require(f.file_size <= 12 * 1024 * 1024, f"oversized individual result {f.filename}")
    require(z.testzip() is None, "corrupted reviewer ZIP")
    manifest = json.loads(z.read("MANIFEST.json"))
    require(manifest["format"] == "navhints-sanitized-summary-v1", "unknown evidence format")
    require(isinstance(manifest.get("public_release_approved"), bool), "release status must be explicit")
    sources = manifest["sources"]
    require(len(sources) >= len(public_digests), "review ZIP missing recorded experiments")
    retain = {}
    source_digests = set()
    for source in sources:
        digest = source["original_zip_sha256"]
        require(is_hash(digest, 64), "malformed original artifact checksum")
        require(digest not in source_digests, "duplicate original source ZIP")
        source_digests.add(digest)
        for item in source["retained"]:
            name = item["published_member"]
            require(name.startswith("evidence/") and name.endswith(".json"), f"unexpected result path: {name}")
            require(name in names and name not in retain, f"missing or duplicated result: {name}")
            payload = z.read(name)
            require(len(payload) == item["sanitized_bytes"], f"{name}: wrong length")
            require(hashlib.sha256(payload).hexdigest() == item["sanitized_sha256"], f"{name}: checksum mismatch")
            json.loads(payload)
            retain[name] = True
    require(public_digests.keys() <= source_digests, "review ZIP omits publicly manifested sources")
    require(len(retain) == manifest["results_members"], "result count does not match manifest")
    require(all(name in retain for name in names if name.startswith("evidence/")),
            "unmanifested result payload found")
    return manifest


def curve(rows: dict) -> list[tuple[float, float]]:
    return sorted([(float(v["recall_percent"]), float(v["mean_ios"])) for v in rows.values()])


def interpolate(points: list[tuple[float, float]], target: float) -> float | None:
    if target < points[0][0] - 1e-7 or target > points[-1][0] + 1e-7:
        return None
    for a, b in points:
        if abs(a - target) < 1e-7:
            return b
    for (ar, ac), (br, bc) in zip(points, points[1:]):
        if ar <= target <= br and br > ar:
            return ac + (target - ar) / (br - ar) * (bc - ac)
    return None


def approx(actual: float, expected: float, limit: float, label: str) -> None:
    require(math.isfinite(actual) and abs(actual - expected) <= limit,
            f"{label}: expected {expected:.6f}, observed {actual:.6f}")


def primary(z: zipfile.ZipFile, summary: dict) -> None:
    lookup = {x["dataset"]: x for x in summary["headline_results"]}
    for key, (label, basename) in DATASETS.items():
        result = json.loads(z.read(f"evidence/{key}/{basename}"))
        base = curve(result["summary"]["baseline"])
        core = curve(result["summary"]["core"])
        per_target = []
        for r, cost in core:
            baseline = interpolate(base, r)
            if baseline is not None:
                per_target.append(100 * (1 - cost / baseline))
        target_count = {"coveo": 4, "pubmed": 5, "text2image": 5, "bigann": 4}[key]
        require(len(per_target) == target_count, f"{key}: wrong common-recall target count")
        saving = statistics.mean(per_target)
        approx(saving, lookup[label]["read_savings_pct"], .0001, f"{key} SSD read saving")
        print(f"  {key:12s} {target_count} matched targets, Core SSD read saving {saving:9.5f}%  PASS")


def pareto(points: list[dict]) -> list[tuple[float, float]]:
    # Same documented per-(seed,repetition) nondominance rule as the manuscript.
    return sorted([(a["recall"], a["mean_service_us"]) for a in points
        if not any(b["recall"] >= a["recall"] - 1e-8 and
                   b["mean_service_us"] <= a["mean_service_us"] + 1e-8 and
                   (b["recall"] > a["recall"] + 1e-8 or
                    b["mean_service_us"] < a["mean_service_us"] - 1e-8)
                   for b in points)])


def latency(z: zipfile.ZipFile, summary: dict) -> None:
    lookup = {x["dataset"]: x for x in summary["headline_results"]}
    for key, (basename, wanted_targets) in SERVICE.items():
        data = json.loads(z.read(f"evidence/service-{key}/{basename}"))
        if key == "pubmed":
            widths, seeds, reps = data["widths"], data["catapult_seeds"], data["repetitions"]
        else:
            widths = data["protocol"]["widths"]
            seeds, reps = data["catapult"]["seeds"], data["protocol"]["rotated_repetitions"]
        require(reps == 3 and sorted(seeds) == [0, 1, 2], f"{key}: nonmatching replay protocol")
        measurements = data["measurements"]
        def points(rep: int, mode: str) -> list[tuple[float, float]]:
            return pareto([measurements[f"r{rep}-L{L}-{mode}"] for L in widths])
        matched = []
        for L in widths:
            target = statistics.mean(measurements[f"r{rep}-L{L}-baseline"]["recall"] for rep in range(reps))
            costs = {}
            for arm in ("baseline", "navhints", "catapult"):
                measurements_at_recall = []
                for rep in range(reps):
                    for seed in (seeds if arm == "catapult" else [None]):
                        actual = "catapult-s" + str(seed) if arm == "catapult" else arm
                        value = interpolate(points(rep, actual), target)
                        if value is not None:
                            measurements_at_recall.append(value)
                require_or_skip = reps * (len(seeds) if arm == "catapult" else 1)
                if len(measurements_at_recall) != require_or_skip:
                    break
                costs[arm] = statistics.mean(measurements_at_recall)
            if len(costs) == 3:
                matched.append((100 * (1 - costs["navhints"] / costs["baseline"]),
                                100 * (1 - costs["navhints"] / costs["catapult"])))
        require(len(matched) == wanted_targets, f"{key}: unsupported recall extrapolation or missing target")
        vs_diskann = statistics.mean(x[0] for x in matched)
        vs_catapult = statistics.mean(x[1] for x in matched)
        name = DATASETS[key][0]
        approx(vs_diskann, lookup[name]["service_latency_vs_diskann_pct"], .006,
               f"{key}: complete-request latency vs DiskANN")
        approx(vs_catapult, lookup[name]["service_latency_vs_catapult_routing_pct"], .006,
               f"{key}: complete-request latency vs Catapult routing")
        print(f"  {key:12s} {wanted_targets} matched targets, latency vs DiskANN {vs_diskann:8.4f}%,"
              f" vs Catapult {vs_catapult:8.4f}%  PASS")


def newest_ablations(z: zipfile.ZipFile, summary: dict) -> None:
    fig = summary["figure8"]
    warm = json.loads(z.read("evidence/figure8-warmup-k1-k2-k4-k8/final-core-warmup-starts.json"))
    require(warm["history_prefixes"] == fig["history_queries"], "Figure 8(a) wrong history positions")
    for i, sample in enumerate(warm["windows"]):
        for k in STARTS:
            actual = sample["starts"][str(k)]["mean_io_saving_percent"]
            approx(actual, fig["warmup"]["k" + str(k)][i], .000002, f"Figure 8(a) history={i},k={k}")
    repeat = json.loads(z.read("evidence/figure8-locality-k1-k2-k4-k8/final-core-starts.json"))
    for i, pool in enumerate(repeat["pools"]):
        approx(pool["requests_per_identity"], fig["replay_counts"][i], 1e-9,
               f"Figure 8(b) working set {i}")
        for k in STARTS:
            actual = pool["starts"][str(k)]["vs_junctions"]["io_saving_percent"]
            approx(actual, fig["locality"]["k" + str(k)][i], .000002,
                   f"Figure 8(b) working set={i},k={k}")
    print("  Figure 8    all 28 warm-up and 20 repetition observations  PASS")

    comparisons = {x["dataset"]: x for x in summary["junction_scoring"]["workloads"]}
    ids = {"coveo": "Coveo", "pubmed": "PubMed", "text2image": "Text2Image-10M", "bigann": "BigANN-10M"}
    for key, name in ids.items():
        data = json.loads(z.read(f"evidence/junction-score-{key}/junction-score-ablation.json"))
        expected = comparisons[name]
        require(data["controller"]["B"] == expected["budget"], f"{key}: unexpected B budget")
        rows = data["matched"]
        require(len(rows) == expected["common_targets"], f"{key}: wrong ranking target count")
        from_frequency = statistics.mean(x["sum_vs_count_saving_percent"] for x in rows)
        from_mean = statistics.mean(x["sum_vs_mean_saving_percent"] for x in rows)
        approx(from_frequency, expected["vs_visit_frequency_pct"], .006,
               f"{key} score vs frequency")
        approx(from_mean, expected["vs_mean_position_pct"], .006,
               f"{key} score vs mean")
        print(f"  {key:12s} ranking B={expected['budget']}, {len(rows)} targets," 
              f" vs visits {from_frequency:7.4f}%, vs mean {from_mean:7.4f}%  PASS")


def destination_controls(z: zipfile.ZipFile) -> list[dict]:
    """Recompute Table 5 from measured recall/read frontiers, not reported gains."""
    targets = (35., 45., 55., 65., 72., 75.)
    source = json.loads(z.read("evidence/destination-equal-work/demand-vs-random.json"))
    capacity = json.loads(z.read("evidence/destination-capacity/reviewer-defense.json"))
    require(source["evaluation"] == {
        "Ls": [10, 20, 40, 80, 160, 320], "reps": 3,
        "replay_queries": 5000, "warmup": 4000, "measured": 1000,
    }, "Table 5: unexpected source-control replay protocol")
    require(capacity["controller_blob"] == "2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb",
            "Table 5: unexpected historical controller")

    def saving(summaries: dict, reference: str, arm: str) -> float:
        baseline, candidate = curve(summaries[reference]), curve(summaries[arm])
        values = []
        for target in targets:
            b, c = interpolate(baseline, target), interpolate(candidate, target)
            require(b is not None and c is not None and b > 0,
                    f"Table 5 {arm}: target {target} outside common recall support")
            values.append(100 * (1 - c / b))
        return statistics.mean(values)

    rows = []
    for arm, label, published in (
        ("randomdb512", "Random corpus IDs", -.90),
        ("history512", "Reservoir of past destinations", 4.41),
        ("recent512", "Recent512 (unique FIFO)", 4.78),
    ):
        observed = saving(source["summary"], "entry", arm)
        approx(observed, published, .005, "Table 5 " + label)
        recorded = next(r for r in source["ranking"] if r["method"] == arm)
        approx(observed, -recorded["mean_io_change_percent_vs_entry"], 1e-9,
               "Table 5 source recorded summary")
        rows.append({"block": "equal-work", "configuration": label,
                     "capacity_ids": 512, "read_savings_pct": observed})
    require([r["capacity_ids"] for r in capacity["capacity_sweep"]]
            == [0, 64, 128, 256, 512, 1024, 2048], "Table 5: incomplete capacity sweep")
    for size, published in ((64, 1.64), (128, 2.77), (256, 3.50),
                            (512, 4.78), (1024, 5.17), (2048, 5.26)):
        observed = saving(capacity["summary"], "cap0", f"cap{size}")
        approx(observed, published, .005, f"Table 5 capacity {size}")
        recorded = next(r for r in capacity["capacity_sweep"] if r["capacity_ids"] == size)
        require(recorded["vs_entry"]["points"] == len(targets), "Table 5 missing capacity targets")
        approx(observed, recorded["vs_entry"]["mean_io_saving_percent"], 1e-9,
               "Table 5 capacity recorded summary")
        rows.append({"block": "capacity", "configuration": f"{size} IDs",
                     "capacity_ids": size, "read_savings_pct": observed})
    approx(rows[2]["read_savings_pct"], rows[6]["read_savings_pct"], 1e-9,
           "Table 5 independent Recent512 controls")
    print("  Table 5     all 3 source controls and 6 capacities, six recall targets each  PASS")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, help="Data-minimized reviewer ZIP (no network required)")
    args = ap.parse_args()
    public_digests = check_source_manifests()
    print(f"Public source inventories: {len(public_digests)} original archive checksums (14 + 9 + 7 + 2)  PASS")
    if args.bundle is None:
        print("Manifest-only preflight completed; supply --bundle to reproduce paper results.")
        return 0
    summary = json.loads((ROOT / "repro/current-paper-summary-20261017.json").read_text())
    with zipfile.ZipFile(args.bundle) as z:
        meta = archive_checks(z, public_digests)
        print(f"Review ZIP: {len(meta['sources'])} source archives, {meta['results_members']}"
              " checksum-verified aggregate JSON outputs  PASS")
        primary(z, summary)
        latency(z, summary)
        newest_ablations(z, summary)
        destination_controls(z)
    print("PASS: NavHints data-minimized manuscript reproduction completed without SSD work")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, IndexError, zipfile.BadZipFile, json.JSONDecodeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
