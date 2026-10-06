#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/catapult-vs-progressive"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART"

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
HELD_GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
TRAIN_GT="$HOME/.cache/geoivf/catapult-paper-pubmed1m/throughput-only.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
IVF16="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/canonical-b16000-c512.bin"
for f in "$QUERIES" "$HELD_GT" "$TRAIN_GT" "$IVF16"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/patch_diskann_catapult_snapshot.py   scripts/patch_diskann_progressive_navhints_k16_clean.py   scripts/qualify_catapult_vs_progressive_navhints.py

export CARGO_HOME="$RUNNER_TEMP/catnav-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/catnav-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/catnav-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/catnav-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-cat-current third_party/DiskANN-nav-current

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-cat-current
git -C third_party/DiskANN-cat-current checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_catapult_snapshot.py third_party/DiskANN-cat-current
(cd third_party/DiskANN-cat-current && cargo fmt --all)
git -C third_party/DiskANN-cat-current diff --check
git -C third_party/DiskANN-cat-current diff > "$ART/catapult.patch"
(cd third_party/DiskANN-cat-current && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/catapult-build.log"
CAT_BIN="$PWD/third_party/DiskANN-cat-current/target/release/diskann-benchmark"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-nav-current
git -C third_party/DiskANN-nav-current checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_progressive_navhints_k16_clean.py third_party/DiskANN-nav-current
(cd third_party/DiskANN-nav-current && cargo fmt --all)
git -C third_party/DiskANN-nav-current diff --check
git -C third_party/DiskANN-nav-current diff > "$ART/navhints.patch"
(cd third_party/DiskANN-nav-current && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/navhints-build.log"
NAV_BIN="$PWD/third_party/DiskANN-nav-current/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/catnav-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_catapult_vs_progressive_navhints.py   --catapult-binary "$CAT_BIN" --nav-binary "$NAV_BIN"   --queries "$QUERIES" --heldout-gt "$HELD_GT" --training-gt "$TRAIN_GT"   --index-prefix "$INDEX_PREFIX" --ivf-16k "$IVF16"   --work "$WORK" --out "$OUT"   2>&1 | tee "$ART/eval.log"
rm -rf "$WORK"
