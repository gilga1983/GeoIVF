#!/usr/bin/env python3
"""Prepare all PubMed paper-ablation Hint-IVF states deterministically."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    subprocess.run([str(x) for x in cmd], check=True)


def load_manifest(path: Path):
    return json.loads(path.read_text())


def prepare_state(
    *,
    base: Path,
    landmark_file: Path,
    state_dir: Path,
    name: str,
    kind: str,
    variant: str,
    history_rows: int,
    requested_budget: int,
    actual_ids: int,
):
    state_dir.mkdir(parents=True, exist_ok=True)
    lm = state_dir / "landmarks.bin"
    shutil.copy2(landmark_file, lm)
    ivf = state_dir / "ivf.bin"
    run([
        sys.executable,
        Path(__file__).with_name("build_hint_ivf.py"),
        "--base", base,
        "--hints", lm,
        "--out", ivf,
        "--nlist", "512",
        "--iterations", "5",
        "--batch", "2048",
        "--seed", "20261005",
    ])
    runtime_extra = 512 * (64 + 4)  # packed coarse 64-byte PQ code + local u32 ID
    meta = {
        "name": name,
        "kind": kind,
        "variant": variant,
        "history_rows": history_rows,
        "requested_budget": requested_budget,
        "actual_ids": actual_ids,
        "ivf_persistent_bytes": ivf.stat().st_size,
        "packed_coarse_runtime_extra_bytes": runtime_extra,
        "runtime_total_payload_bytes": ivf.stat().st_size + runtime_extra,
    }
    (state_dir / "state.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for n in ("base", "trace", "work", "out"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    learner = Path(__file__).with_name("learn_landmark_variants.py")

    # Scoring ablation: exactly one 5K history and nominal 16K budget.
    score_work = args.work / "scoring"
    score_work.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, learner,
        "--trace", args.trace,
        "--out-dir", score_work,
        "--history-rows", "5000",
        "--budgets", "16000",
        "--variants", "skip,support,mean-pos,random,destination",
    ])
    sm = load_manifest(score_work / "variants-h5000.manifest.json")
    for key, item in sm["emitted"].items():
        variant = item["variant"]
        # Keep the destination baseline even though it naturally has fewer
        # than 16K unique historical results; all other scoring arms must fill 16K.
        if variant != "destination" and not item["budget_saturated"]:
            raise RuntimeError(f"scoring variant {variant} failed to fill 16K")
        prepare_state(
            base=args.base,
            landmark_file=score_work / item["file"],
            state_dir=args.out / "scoring" / variant,
            name=f"score-{variant}",
            kind="scoring",
            variant=variant,
            history_rows=5000,
            requested_budget=16000,
            actual_ids=item["actual_ids"],
        )

    # Training-size sweep. Fixed capacities are reported only when saturated.
    # This prevents training amount from silently changing memory capacity.
    histories = (250, 500, 1000, 2500, 5000)
    budgets = (3000, 4096, 16000)
    train_eligibility = {}
    for h in histories:
        hw = args.work / f"train-h{h}"
        hw.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, learner,
            "--trace", args.trace,
            "--out-dir", hw,
            "--history-rows", str(h),
            "--budgets", ",".join(map(str, budgets)),
            "--variants", "skip",
        ])
        manifest = load_manifest(hw / f"variants-h{h}.manifest.json")
        train_eligibility[str(h)] = manifest["eligible_traversal_vertices"]
        for budget in budgets:
            item = manifest["emitted"].get(f"skip-b{budget}")
            if item is None or not item["budget_saturated"]:
                continue
            prepare_state(
                base=args.base,
                landmark_file=hw / item["file"],
                state_dir=args.out / "training" / f"h{h}-b{budget}",
                name=f"train-h{h}-b{budget}",
                kind="training-size",
                variant="skip",
                history_rows=h,
                requested_budget=budget,
                actual_ids=item["actual_ids"],
            )

    result = {
        "scoring_manifest": sm,
        "training_eligible_vertices": train_eligibility,
        "training_histories": list(histories),
        "fixed_budgets": list(budgets),
        "rule": "training points emitted only when the fixed budget is fully populated",
    }
    (args.out / "prepared-states.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
