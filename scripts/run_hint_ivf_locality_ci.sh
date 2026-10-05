#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-ivf-locality

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-ivf-locality-v1"
LANDMARK_DIR="$ROOT/landmarks"
ORIG="$ROOT/hints-b16000-nlist512-spherical.bin"
LOCAL="$ROOT/hints-b16000-nlist512-spherical-locality.bin"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_DIR"
git rev-parse HEAD > artifacts/hint-ivf-locality/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-ivf-locality/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/reorder_hint_ivf_locality.py   scripts/patch_diskann_hint_ivf_hybrid.py   scripts/qualify_hint_ivf_locality.py

.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "16000"   2>&1 | tee artifacts/hint-ivf-locality/learn.log

.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE"   --hints "$LANDMARK_DIR/landmarks-b16000.bin"   --out "$ORIG"   --nlist 512   --iterations 5   --batch 2048   --seed 20261005   2>&1 | tee artifacts/hint-ivf-locality/build-ivf.log

.venv/bin/python scripts/reorder_hint_ivf_locality.py   --input "$ORIG"   --output "$LOCAL"   2>&1 | tee artifacts/hint-ivf-locality/reorder.log
cp "$ORIG.manifest.json" artifacts/hint-ivf-locality/original.manifest.json
cp "$LOCAL.manifest.json" artifacts/hint-ivf-locality/locality.manifest.json

export CARGO_HOME="$RUNNER_TEMP/hintlocal-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintlocal-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintlocal-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintlocal-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hint-locality
git -C third_party/DiskANN-hint-locality checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_hybrid.py third_party/DiskANN-hint-locality
(
  cd third_party/DiskANN-hint-locality
  cargo fmt --all
  cargo fmt --all -- --check
  cargo check --release --locked -p diskann-benchmark --features disk-index
  cargo build --release --locked -p diskann-benchmark --features disk-index
) 2>&1 | tee artifacts/hint-ivf-locality/build.log
BIN="$PWD/third_party/DiskANN-hint-locality/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintlocal-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-ivf-locality/results
source scripts/perf_guard.sh
geoivf_perf_lock
.venv/bin/python scripts/qualify_hint_ivf_locality.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --original-ivf "$ORIG"   --locality-ivf "$LOCAL"   --work "$WORK"   --out "$PWD/artifacts/hint-ivf-locality/results"   --reps 7   2>&1 | tee artifacts/hint-ivf-locality/eval.log

rm -rf "$WORK"
