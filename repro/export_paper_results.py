#!/usr/bin/env python3
"""Regenerate NavHints plot data from bundled aggregates, without a private repo.

  python3 repro/export_paper_results.py --output artifacts/paper
  cd artifacts/paper && pdflatex -interaction=nonstopmode -halt-on-error figures.tex

Python standard library only. No input dataset downloads, ANN builds or SSD
measurements. Original plotting/derivation sources are in repro/paper/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import zipfile
from pathlib import Path

import verify_review_bundle as check

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "repro/paper"
DATASET_LABELS = {
    "coveo": "Coveo",
    "pubmed": "PubMed",
    "text2image": "Text2Image-10M",
    "bigann": "BigANN-10M",
}


def load_paper_module(name: str):
    """Load an immutable manuscript derivation without changing its source."""
    spec = importlib.util.spec_from_file_location(
        name, PAPER / "scripts" / (name + ".py")
    )
    check.require(
        spec is not None and spec.loader is not None,
        f"Cannot load pinned paper derivation: {name}",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_csv(directory: Path, name: str, fields: list[str], rows: list[dict]) -> None:
    """Write stable column order and newline conventions for the TeX plots."""
    with (directory / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_reference_csvs(directory: Path) -> int:
    """Compare column order, cell presence and numeric values with the paper.

    Matched NaNs denote the manuscript's intentionally unsupported Pareto
    points. Numeric cells retain the original tolerance for rounded CSV data.
    """
    reference_files = sorted((PAPER / "reference-data").glob("*.csv"))
    check.require(bool(reference_files), f"No frozen reference CSVs under {PAPER}")
    for reference in reference_files:
        with reference.open(newline="") as stream:
            reader = csv.DictReader(stream)
            expected_columns = reader.fieldnames
            expected_rows = list(reader)
        with (directory / reference.name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            actual_columns = reader.fieldnames
            actual_rows = list(reader)
        check.require(
            actual_columns == expected_columns,
            f"{reference.name}: column names or order differ from the manuscript",
        )
        check.require(
            len(actual_rows) == len(expected_rows), f"{reference.name}: wrong row count"
        )
        for row_number, (actual, expected) in enumerate(
            zip(actual_rows, expected_rows), start=2
        ):
            check.require(
                actual.keys() == expected.keys(),
                f"{reference.name}: unexpected cells in row {row_number}",
            )
            for column in expected:
                observed_cell, expected_cell = actual[column], expected[column]
                label = f"{reference.name}: row {row_number}, column {column}"
                check.require(
                    observed_cell is not None and expected_cell is not None,
                    label + ": missing cell",
                )
                try:
                    observed_value, expected_value = float(observed_cell), float(
                        expected_cell
                    )
                except ValueError:
                    check.require(
                        observed_cell == expected_cell, label + ": text mismatch"
                    )
                else:
                    both_missing = math.isnan(observed_value) and math.isnan(
                        expected_value
                    )
                    same_value = math.isclose(
                        observed_value, expected_value, rel_tol=0, abs_tol=0.00011
                    )
                    check.require(
                        both_missing or same_value,
                        f"{label}: {observed_value} differs from manuscript {expected_value}",
                    )
    count = len(reference_files)
    print(
        f"  Paper CSVs  {count} regenerated files match the submitted reference values  PASS"
    )
    return count


def read_evidence(archive: zipfile.ZipFile, member: str) -> dict:
    """Read one checksum-verified aggregate using its evidence-relative path."""
    return json.loads(archive.read("evidence/" + member))


def export_read_frontiers(archive: zipfile.ZipFile, data: Path) -> None:
    """Regenerate the read-saving bars and all measured recall/read curves."""
    front = load_paper_module("derive_final_core_eval")
    components = []
    for idx, key in enumerate(check.DATASETS):
        raw = read_evidence(archive, key + "/" + check.DATASETS[key][1])
        curves = {
            m: front.frontier(raw["summary"][m]) for m in ("baseline", "ivf", "core")
        }
        matched = []
        for _, point in sorted(raw["summary"]["core"].items(), key=lambda x: int(x[0])):
            recall = float(point["recall_percent"])
            at = {m: front.interpolate(c, recall) for m, c in curves.items()}
            if any(v is None for v in at.values()):
                continue
            baseline_reads, junction_reads, navhints_reads = [
                at[method]["reads"] for method in ("baseline", "ivf", "core")
            ]
            matched.append(
                {
                    "recall": recall,
                    "junction": 100 * (1 - junction_reads / baseline_reads),
                    "navhints": 100 * (1 - navhints_reads / baseline_reads),
                }
            )
        components.append(
            {
                "dataset_id": idx,
                "dataset": DATASET_LABELS[key],
                **{
                    col: round(statistics.mean(p[col] for p in matched), 4)
                    for col in ("junction", "navhints")
                },
                "anchors": len(matched),
                "recall_min": round(min(p["recall"] for p in matched), 4),
                "recall_max": round(max(p["recall"] for p in matched), 4),
            }
        )
        for method, curve in curves.items():
            name = "junction" if method == "ivf" else method
            write_csv(
                data,
                f"eval_frontier_{key}_{name}.csv",
                ["recall", "ssd_reads", "latency_us"],
                [
                    {
                        "recall": round(p["recall"], 6),
                        "ssd_reads": round(p["reads"], 6),
                        "latency_us": round(p["latency"], 4),
                    }
                    for p in curve
                ],
            )
    write_csv(data, "eval_components.csv", list(components[0]), components)


def export_service_latency(archive: zipfile.ZipFile, output: Path, data: Path) -> None:
    """Regenerate complete-request curves using the pinned analysis rules."""
    service = load_paper_module("derive_service_latency")
    latency = []
    for key, dataset, filename, label in service.DATASETS:
        raw = read_evidence(archive, "service-" + key + "/" + check.SERVICE[key][0])
        result = service.calculate(raw, dataset)
        service.write_csv(data / filename, result)
        result.pop("plot")
        latency.append({"label": label, "result": result})
    (output / "service-latency.json").write_text(json.dumps(latency, indent=2) + "\n")


def export_start_sensitivity(archive: zipfile.ZipFile, data: Path) -> None:
    """Export the warm-up and repetition panels at their original start counts."""
    warm = read_evidence(
        archive, "figure8-warmup-k1-k2-k4-k8/final-core-warmup-starts.json"
    )
    rows = [
        {
            "history": row["history_queries"],
            **{
                f"k{k}": row["starts"][str(k)]["mean_io_saving_percent"]
                for k in check.STARTS
            },
        }
        for row in warm["windows"]
    ]
    write_csv(data, "eval_warmup_starts_final_core.csv", list(rows[0]), rows)
    locality = read_evidence(
        archive, "figure8-locality-k1-k2-k4-k8/final-core-starts.json"
    )
    rows = [
        {
            "repeats": p["requests_per_identity"],
            **{
                f"k{k}": p["starts"][str(k)]["vs_junctions"]["io_saving_percent"]
                for k in check.STARTS
            },
        }
        for p in locality["pools"]
    ]
    write_csv(data, "eval_locality_starts_final_core.csv", list(rows[0]), rows)


def export_motivation(archive: zipfile.ZipFile, data: Path) -> None:
    """Export held-out junction concentration and causal destination overlap."""
    overlap = {}
    for key, source in (
        ("pubmed", "pubmed1M"),
        ("coveo", "coveo"),
        ("text2image", "text2image-10M"),
        ("bigann", "bigann-10M"),
    ):
        raw = read_evidence(archive, "motivation-destination-reuse/" + source + ".json")
        overlap[key] = {
            r["capacity"]: r["any_exact_top10_present_pct"] for r in raw["results"]
        }
        junction = read_evidence(
            archive, "motivation-junction-concentration/" + source + ".json"
        )
        write_csv(
            data,
            "motivation_junctions_" + key + ".csv",
            ["corpus_percent", "heldout_visits_percent", "unseen_ids"],
            [
                {
                    "corpus_percent": 100 * r["corpus_fraction"],
                    "heldout_visits_percent": 100 * r["heldout_visit_share"],
                    "unseen_ids": r["ids_not_seen_in_training"],
                }
                for r in junction["curve"]
            ],
        )
    write_csv(
        data,
        "recent_destination_overlap.csv",
        ["capacity", *overlap],
        [
            {"capacity": c, **{key: vals[c] for key, vals in overlap.items()}}
            for c in (0, 16, 32, 64, 128, 256, 512, 1024, 2048)
        ],
    )


def export_controls(
    archive: zipfile.ZipFile, data: Path, destination_rows: list[dict]
) -> None:
    """Export independently checked destination controls and junction scoring."""
    write_csv(
        data,
        "table5_destination_controls.csv",
        list(destination_rows[0]),
        destination_rows,
    )
    scoring = []
    for key in check.DATASETS:
        raw = read_evidence(
            archive, "junction-score-" + key + "/junction-score-ablation.json"
        )
        scoring.append(
            {
                "dataset": key,
                "junctions": raw["controller"]["B"],
                "targets": len(raw["matched"]),
                "vs_visits_pct": statistics.mean(
                    r["sum_vs_count_saving_percent"] for r in raw["matched"]
                ),
                "vs_mean_position_pct": statistics.mean(
                    r["sum_vs_mean_saving_percent"] for r in raw["matched"]
                ),
            }
        )
    write_csv(data, "table7_junction_scoring.csv", list(scoring[0]), scoring)


def export_paper_data(
    archive: zipfile.ZipFile, output: Path, destination_rows: list[dict]
) -> int:
    """Write plot inputs in paper order, then compare all frozen references."""
    data = output / "figures/data"
    data.mkdir(parents=True, exist_ok=True)
    export_read_frontiers(archive, data)
    export_service_latency(archive, output, data)
    export_start_sensitivity(archive, data)
    export_motivation(archive, data)
    export_controls(archive, data, destination_rows)
    return compare_reference_csvs(data)


def copy_paper_sources(output: Path) -> None:
    """Copy portable source assets, excluding interpreter caches from earlier runs."""
    for directory in ("figures", "tables", "scripts"):
        shutil.copytree(
            PAPER / directory,
            output / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("macros.tex", "figures.tex", "SOURCE.json"):
        shutil.copyfile(PAPER / name, output / name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "repro/navhints-reviewer-evidence.zip"
    )
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/paper")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files in an existing output directory",
    )
    a = p.parse_args()
    release = json.loads((ROOT / "repro/release-20261025.json").read_text())
    check.require(
        hashlib.sha256(a.bundle.read_bytes()).hexdigest() == release["sha256"],
        "Evidence ZIP differs from the frozen public release checksum",
    )
    # Check every pinned reference/plot source before using it.
    assets = json.loads((PAPER / "SOURCE.json").read_text())
    for item in assets["files"]:
        check.require(
            hashlib.sha256((PAPER / item["path"]).read_bytes()).hexdigest()
            == item["sha256"],
            "Paper source snapshot changed: " + item["path"],
        )
    summary = json.loads(
        (ROOT / "repro/current-paper-summary-20261017.json").read_text()
    )
    with zipfile.ZipFile(a.bundle) as z:
        check.archive_checks(z, check.check_source_manifests())
        check.primary(z, summary)
        check.latency(z, summary)
        check.newest_ablations(z, summary)
        destination_rows = check.destination_controls(z)
        check.require(
            not a.output.exists() or not any(a.output.iterdir()) or a.overwrite,
            "Output directory is nonempty; choose a new path or pass --overwrite",
        )
        a.output.mkdir(parents=True, exist_ok=True)
        # Static manuscript plot/table sources are copied exactly. Numeric plot
        # CSVs are independently regenerated below, not copied from references.
        copy_paper_sources(a.output)
        count = export_paper_data(z, a.output, destination_rows)
    (a.output / "REPRODUCTION.json").write_text(
        json.dumps(
            {
                "evidence_sha256": release["sha256"],
                "paper_commit": assets["paper_commit"],
                "regenerated_reference_csvs": count,
                "ssd_measurements_rerun": False,
                "tables": "Pinned manuscript TeX tables; numerical verification coverage is listed in ARTIFACT_MAP.md",
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS: standalone paper plotting directory prepared:", a.output)
    print(
        "Optional figure/table rendering: cd",
        a.output,
        "&& pdflatex figures.tex && pdflatex figures.tex",
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
        raise SystemExit("FAIL: " + str(e))
