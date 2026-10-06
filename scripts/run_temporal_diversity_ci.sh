#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/temporal-diversity"
ROOT="$HOME/.cache/geoivf/temporal-diversity-v1"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART" "$ROOT"

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L64.jsonl"
GENERAL8="$HOME/.cache/geoivf/paper-memory-frontier-v1/landmarks/landmarks-b8192.bin"
CANONICAL="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/canonical-b16000-c512.bin"
for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$GENERAL8" "$CANONICAL"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/learn_temporal_diverse_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_progressive_navhints.py   scripts/qualify_temporal_diversity.py

IDS="$ROOT/temporal-8k-4k-2k-2k.bin"
IVF="$ROOT/temporal-b16384-c512.bin"
.venv/bin/python scripts/learn_temporal_diverse_landmarks.py   --trace "$TRACE" --general-ids "$GENERAL8" --out "$IDS"   --stages "8,16,24" --budgets "4096,2048,2048"   2>&1 | tee "$ART/learn.log"
cp "$IDS.manifest.json" "$ART/temporal-vocabulary.manifest.json"

.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE" --hints "$IDS" --out "$IVF"   --nlist 512 --iterations 5 --batch 2048 --seed 20261006   2>&1 | tee "$ART/build-ivf.log"
cp "$IVF.manifest.json" "$ART/temporal-ivf.manifest.json"
sha256sum "$IDS" "$IVF" > "$ART/state.sha256"

export CARGO_HOME="$RUNNER_TEMP/temporal-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/temporal-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/temporal-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/temporal-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-temporal-diverse
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-temporal-diverse
git -C third_party/DiskANN-temporal-diverse checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_progressive_navhints.py third_party/DiskANN-temporal-diverse
(cd third_party/DiskANN-temporal-diverse && cargo fmt --all)
git -C third_party/DiskANN-temporal-diverse diff --check
(cd third_party/DiskANN-temporal-diverse && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-temporal-diverse/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/temporal-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_temporal_diversity.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --ivf-canonical "$CANONICAL" --ivf-temporal "$IVF"   --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
