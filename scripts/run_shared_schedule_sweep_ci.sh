#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/shared-schedule-sweep"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART"

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
CANON_IVF="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/canonical-b16000-c512.bin"
START_IVF="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/stage0-b8000-c256.bin"
RESIDUAL_IVF="$HOME/.cache/geoivf/shared-multistage-navhints-v1/residual-stage3-b4096-c128.bin"
for f in "$QUERIES" "$GT" "$CANON_IVF" "$START_IVF" "$RESIDUAL_IVF"; do test -s "$f"; done

.venv/bin/python -m py_compile   scripts/patch_diskann_shared_multistage_navhints.py   scripts/qualify_staged_navhints.py   scripts/qualify_shared_schedule_sweep.py

export CARGO_HOME="$RUNNER_TEMP/sharedsweep-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/sharedsweep-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/sharedsweep-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/sharedsweep-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-shared-sweep
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-shared-sweep
git -C third_party/DiskANN-shared-sweep checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_shared_multistage_navhints.py third_party/DiskANN-shared-sweep
(cd third_party/DiskANN-shared-sweep && cargo fmt --all)
git -C third_party/DiskANN-shared-sweep diff --check
(cd third_party/DiskANN-shared-sweep && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-shared-sweep/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/sharedsweep-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_shared_schedule_sweep.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --ivf-16k "$CANON_IVF" --ivf-start8k "$START_IVF" --ivf-residual4k "$RESIDUAL_IVF"   --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"
rm -rf "$WORK"
