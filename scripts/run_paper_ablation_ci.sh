#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
ROOT="$HOME/.cache/geoivf/paper-ablation-v1"
STATE_ROOT="$ROOT/states"
WORK="$RUNNER_TEMP/paper-ablation-$GITHUB_RUN_ID"
ART="$PWD/artifacts/paper-ablation"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE"; do test -s "$f"; done
rm -rf "$WORK" "$ART"
mkdir -p "$WORK" "$ART" "$STATE_ROOT"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"

.venv/bin/python -m py_compile   scripts/learn_landmark_variants.py   scripts/prepare_paper_ablation_states.py   scripts/build_hint_ivf.py   scripts/qualify_paper_ablation.py   scripts/patch_diskann_hint_ivf_packed_direct.py

# Rebuild states deterministically. They are cheap relative to the search campaign
# and this prevents stale learner variants from surviving across commits.
rm -rf "$STATE_ROOT"
.venv/bin/python scripts/prepare_paper_ablation_states.py   --base "$BASE"   --trace "$TRACE"   --work "$WORK/state-work"   --out "$STATE_ROOT"   2>&1 | tee "$ART/prepare.log"
cp "$STATE_ROOT/prepared-states.json" "$ART/"

export CARGO_HOME="$RUNNER_TEMP/paperab-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/paperab-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/paperab-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/paperab-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-paper-ablation
git -C third_party/DiskANN-paper-ablation checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py third_party/DiskANN-paper-ablation
(
  cd third_party/DiskANN-paper-ablation
  cargo fmt --all
  cargo build --release --locked -p diskann-benchmark --features disk-index
) 2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-paper-ablation/target/release/diskann-benchmark"

mkdir -p "$ART/results"
source scripts/perf_guard.sh
geoivf_perf_lock
.venv/bin/python scripts/qualify_paper_ablation.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --state-root "$STATE_ROOT"   --work "$WORK/eval"   --out "$ART/results"   --reps 3   2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
