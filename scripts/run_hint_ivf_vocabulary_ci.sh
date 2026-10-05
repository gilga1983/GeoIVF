#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-ivf-vocabulary

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-ivf-vocabulary-v1"
LANDMARK_ROOT="$ROOT/landmarks"
IVF_ROOT="$ROOT/ivf"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_ROOT" "$IVF_ROOT"
git rev-parse HEAD > artifacts/hint-ivf-vocabulary/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-ivf-vocabulary/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_hint_ivf.py   scripts/qualify_hint_ivf_vocabulary.py

.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_ROOT"   --budgets "16000,20000"   2>&1 | tee artifacts/hint-ivf-vocabulary/learn.log
cp "$LANDMARK_ROOT/global-landmarks.manifest.json" artifacts/hint-ivf-vocabulary/

for budget in 16000 20000; do
  out="$IVF_ROOT/hints-b${budget}-nlist512-spherical.bin"
  .venv/bin/python scripts/build_hint_ivf.py     --base "$BASE"     --hints "$LANDMARK_ROOT/landmarks-b${budget}.bin"     --out "$out"     --nlist 512     --iterations 5     --batch 2048     --seed 20261005     2>&1 | tee "artifacts/hint-ivf-vocabulary/build-b${budget}.log"
  cp "$out.manifest.json"     "artifacts/hint-ivf-vocabulary/b${budget}.manifest.json"
done

export CARGO_HOME="$RUNNER_TEMP/hintvocab-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintvocab-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintvocab-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintvocab-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hintvocab
git -C third_party/DiskANN-hintvocab checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf.py third_party/DiskANN-hintvocab
(cd third_party/DiskANN-hintvocab && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/hint-ivf-vocabulary/build.log
BIN="$PWD/third_party/DiskANN-hintvocab/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintvocab-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-ivf-vocabulary/results
.venv/bin/python scripts/qualify_hint_ivf_vocabulary.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --landmark-root "$LANDMARK_ROOT"   --ivf-root "$IVF_ROOT"   --work "$WORK"   --out "$PWD/artifacts/hint-ivf-vocabulary/results"   --reps 3   2>&1 | tee artifacts/hint-ivf-vocabulary/eval.log

rm -rf "$WORK"
