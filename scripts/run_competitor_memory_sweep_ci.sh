#!/usr/bin/env bash
set -euxo pipefail
DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/competitor-memory-sweep"
mkdir -p "$ART"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
python -m venv .venv
.venv/bin/pip install numpy==2.2.6
.venv/bin/python -m py_compile scripts/qualify_competitor_memory_sweep.py scripts/qualify_frozen_paper_competitors.py scripts/qualify_final_catapult.py scripts/patch_diskann_hot_cache.py

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
TRAIN_GT="$HOME/.cache/geoivf/catapult-paper-pubmed1m/throughput-only.gt"
INDEX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
IVF="$HOME/.cache/geoivf/paper-competitors-phase1-v1/navhints/hints-b16000-nlist512-spherical.bin"
for f in "$BASE" "$QUERIES" "$GT" "$TRAIN_GT" "$TRACE" "$IVF" "$INDEX"_disk.index "$INDEX"_pq_pivots.bin "$INDEX"_pq_compressed.bin; do test -s "$f"; done

HOT_DIR="$RUNNER_TEMP/ram-sweep-hot-$GITHUB_RUN_ID"
mkdir -p "$HOT_DIR"
.venv/bin/python scripts/learn_hot_cache_ids.py --trace "$TRACE" --out-dir "$HOT_DIR" --counts "30,64,128,256,512" > "$ART/hot-cache-learning.json"
for n in 30 64 128 256 512; do test -s "$HOT_DIR/hot-cache-n$n.bin"; done
cp "$HOT_DIR/hot-cache.manifest.json" "$ART/hot-cache-manifest.json"

export CARGO_HOME="$RUNNER_TEMP/ramsweep-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/ramsweep-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/ramsweep-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/ramsweep-rustup-$GITHUB_RUN_ID.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

CACHE_ROOT="$RUNNER_TEMP/ramsweep-diskann-cache-$GITHUB_RUN_ID"
CAT_ROOT="$RUNNER_TEMP/ramsweep-diskann-cat-$GITHUB_RUN_ID"
NAV_ROOT="$RUNNER_TEMP/ramsweep-diskann-nav-$GITHUB_RUN_ID"
for root in "$CACHE_ROOT" "$CAT_ROOT" "$NAV_ROOT"; do
    git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$root"
    git -C "$root" checkout --detach "$DISKANN_REV"
done
.venv/bin/python scripts/patch_diskann_hot_cache.py "$CACHE_ROOT"
.venv/bin/python scripts/patch_diskann_catapult_snapshot.py "$CAT_ROOT"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py "$NAV_ROOT"
.venv/bin/python scripts/patch_diskann_experience_core.py "$NAV_ROOT"

(cd "$CACHE_ROOT"; cargo fmt --all; git diff --check; cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/cache-build.log"
(cd "$CAT_ROOT"; cargo fmt --all; git diff --check; cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/catapult-build.log"
(cd "$NAV_ROOT"; cargo fmt --all; git diff --check; cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/navhints-build.log"

mkdir -p "$ART/results"
.venv/bin/python scripts/qualify_competitor_memory_sweep.py \
    --cache-binary "$CACHE_ROOT/target/release/diskann-benchmark" \
    --catapult-binary "$CAT_ROOT/target/release/diskann-benchmark" \
    --nav-binary "$NAV_ROOT/target/release/diskann-benchmark" \
    --queries "$QUERIES" --heldout-gt "$GT" --training-gt "$TRAIN_GT" \
    --index-prefix "$INDEX" --ivf "$IVF" --hot-dir "$HOT_DIR" \
    --work "$RUNNER_TEMP/ramsweep-eval-$GITHUB_RUN_ID" \
    --out "$ART/results" --reps 3 2>&1 | tee "$ART/eval.log"
