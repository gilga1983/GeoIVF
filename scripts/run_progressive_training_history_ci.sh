#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/progressive-training-history"
ROOT="$HOME/.cache/geoivf/paper-ablation-v1"
STATE_ROOT="$ROOT/states"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART"

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
for f in "$BASE" "$QUERIES" "$GT" "$TRACE"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/patch_diskann_progressive_navhints.py   scripts/qualify_progressive_training_history.py

# Reuse the frozen ablation states when present. Rebuild only if the shared
# cache is missing, preserving the exact learner/state generation used before.
if [ ! -s "$STATE_ROOT/prepared-states.json" ]; then
  mkdir -p "$STATE_ROOT"
  WORK_STATE="$RUNNER_TEMP/progtrain-state-$GITHUB_RUN_ID"
  rm -rf "$WORK_STATE" "$STATE_ROOT"
  mkdir -p "$WORK_STATE" "$STATE_ROOT"
  .venv/bin/python scripts/prepare_paper_ablation_states.py     --base "$BASE" --trace "$TRACE"     --work "$WORK_STATE" --out "$STATE_ROOT"     2>&1 | tee "$ART/prepare-states.log"
  rm -rf "$WORK_STATE"
fi
test -s "$STATE_ROOT/prepared-states.json"
cp "$STATE_ROOT/prepared-states.json" "$ART/"

export CARGO_HOME="$RUNNER_TEMP/progtrain-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/progtrain-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/progtrain-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/progtrain-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-progressive-training
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-progressive-training
git -C third_party/DiskANN-progressive-training checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_progressive_navhints.py third_party/DiskANN-progressive-training
(cd third_party/DiskANN-progressive-training && cargo fmt --all)
git -C third_party/DiskANN-progressive-training diff --check
(cd third_party/DiskANN-progressive-training && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-progressive-training/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/progtrain-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_progressive_training_history.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --state-root "$STATE_ROOT" --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"
rm -rf "$WORK"
