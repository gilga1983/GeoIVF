#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
BUDGETS="2048,4096,8192,16000,20949"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-capacity

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-capacity-v1"
LANDMARK_DIR="$ROOT/landmarks"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_DIR"
git rev-parse HEAD > artifacts/hint-capacity/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-capacity/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_hint_ivf.py   scripts/qualify_hint_capacity.py

.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "$BUDGETS"   2>&1 | tee artifacts/hint-capacity/learn.log
cp "$LANDMARK_DIR/global-landmarks.manifest.json" artifacts/hint-capacity/

# Fixed 256-list arm for every capacity.
for budget in 2048 4096 8192 16000 20949; do
  ivf="$ROOT/hints-b${budget}-nlist256.bin"
  .venv/bin/python scripts/build_hint_ivf.py     --base "$BASE"     --hints "$LANDMARK_DIR/landmarks-b${budget}.bin"     --out "$ivf"     --nlist 256     --iterations 5     --batch 2048     2>&1 | tee "artifacts/hint-capacity/build-fixed-b${budget}.log"
  cp "$ivf.manifest.json" artifacts/hint-capacity/
done

# Capacity-scaled directory: approximately constant hints/list.
for spec in "2048:32" "4096:64" "8192:128" "20949:384"; do
  IFS=: read -r budget nlist <<< "$spec"
  ivf="$ROOT/hints-b${budget}-nlist${nlist}.bin"
  .venv/bin/python scripts/build_hint_ivf.py     --base "$BASE"     --hints "$LANDMARK_DIR/landmarks-b${budget}.bin"     --out "$ivf"     --nlist "$nlist"     --iterations 5     --batch 2048     2>&1 | tee "artifacts/hint-capacity/build-scaled-b${budget}-n${nlist}.log"
  cp "$ivf.manifest.json" artifacts/hint-capacity/
done

export CARGO_HOME="$RUNNER_TEMP/hintcap-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintcap-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintcap-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintcap-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hint-capacity
git -C third_party/DiskANN-hint-capacity checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf.py third_party/DiskANN-hint-capacity
(cd third_party/DiskANN-hint-capacity && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/hint-capacity/build.log
BIN="$PWD/third_party/DiskANN-hint-capacity/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintcap-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-capacity/results
.venv/bin/python scripts/qualify_hint_capacity.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --landmark-dir "$LANDMARK_DIR"   --ivf-dir "$ROOT"   --work "$WORK"   --out "$PWD/artifacts/hint-capacity/results"   --reps 3   2>&1 | tee artifacts/hint-capacity/eval.log

rm -rf "$WORK"
