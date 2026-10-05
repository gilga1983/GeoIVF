#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-ivf

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-ivf-v1"
LANDMARK_DIR="$ROOT/landmarks"
IVF_FILE="$ROOT/hints-b16000-nlist256.bin"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_DIR"
git rev-parse HEAD > artifacts/hint-ivf/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-ivf/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_hint_ivf.py   scripts/qualify_hint_ivf.py

# Learn a flat 2K reference and a much larger 16K dictionary from the same
# ordinary DiskANN L=4 teacher.
.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "2048,16000"   2>&1 | tee artifacts/hint-ivf/learn.log
cp "$LANDMARK_DIR/global-landmarks.manifest.json" artifacts/hint-ivf/

# Build an IVF whose persistent deployment payload remains IDs + offsets only.
.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE"   --hints "$LANDMARK_DIR/landmarks-b16000.bin"   --out "$IVF_FILE"   --nlist 256   --iterations 5   --batch 2048   2>&1 | tee artifacts/hint-ivf/build-ivf.log
cp "$IVF_FILE.manifest.json" artifacts/hint-ivf/

export CARGO_HOME="$RUNNER_TEMP/hintivf-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintivf-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintivf-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintivf-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hint-ivf
git -C third_party/DiskANN-hint-ivf checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf.py third_party/DiskANN-hint-ivf
(cd third_party/DiskANN-hint-ivf && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/hint-ivf/build.log
BIN="$PWD/third_party/DiskANN-hint-ivf/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintivf-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-ivf/results
.venv/bin/python scripts/qualify_hint_ivf.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --flat-2048 "$LANDMARK_DIR/landmarks-b2048.bin"   --flat-16000 "$LANDMARK_DIR/landmarks-b16000.bin"   --hint-ivf "$IVF_FILE"   --work "$WORK"   --out "$PWD/artifacts/hint-ivf/results"   --reps 3   2>&1 | tee artifacts/hint-ivf/eval.log

rm -rf "$WORK"
