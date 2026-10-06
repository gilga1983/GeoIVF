#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/progressive-mechanism"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART"

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
IVF16="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/canonical-b16000-c512.bin"
for f in "$QUERIES" "$GT" "$IVF16"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/patch_diskann_progressive_navhints.py   scripts/qualify_staged_navhints.py   scripts/qualify_progressive_mechanism.py

export CARGO_HOME="$RUNNER_TEMP/progmech-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/progmech-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/progmech-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/progmech-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-progressive-mechanism
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-progressive-mechanism
git -C third_party/DiskANN-progressive-mechanism checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_progressive_navhints.py third_party/DiskANN-progressive-mechanism
(cd third_party/DiskANN-progressive-mechanism && cargo fmt --all)
git -C third_party/DiskANN-progressive-mechanism diff --check
git -C third_party/DiskANN-progressive-mechanism diff > "$ART/progressive-mechanism.patch"
(cd third_party/DiskANN-progressive-mechanism && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-progressive-mechanism/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/progmech-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_progressive_mechanism.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --ivf-16k "$IVF16" --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"
rm -rf "$WORK"
