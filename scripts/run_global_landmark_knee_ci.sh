#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/global-landmark-knee

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
ROUTER="$HOME/.cache/geoivf/region-ablation-routers-v1/nlist-512/router.bin"
WAYPOINT="$HOME/.cache/geoivf/region-ablation-caches-v1/nlist-512/heavy-skip-b2500.bin"
TRACE="$HOME/.cache/geoivf/region-ablation-traces-v1/nlist-512/portal-k1-train5000-trace.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
LANDMARK_DIR="$HOME/.cache/geoivf/global-navigation-landmarks-v1"

for f in "$QUERIES" "$GT" "$ROUTER" "$WAYPOINT" "$TRACE" "$PORTALS"; do test -s "$f"; done
mkdir -p "$LANDMARK_DIR"
git rev-parse HEAD > artifacts/global-landmark-knee/geoivf-commit.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/patch_diskann_global_starts.py   scripts/qualify_global_landmark_knee.py

rm -rf "$LANDMARK_DIR"
mkdir -p "$LANDMARK_DIR"
.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "64,128,256,512,768,1024,1536,2048,2500"   2>&1 | tee artifacts/global-landmark-knee/learn.log
cp "$LANDMARK_DIR/global-landmarks.manifest.json" artifacts/global-landmark-knee/

export CARGO_HOME="$RUNNER_TEMP/landmark-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/landmark-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/landmark-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/landmark-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-landmark-global
git -C third_party/DiskANN-landmark-global checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_global_starts.py third_party/DiskANN-landmark-global
(cd third_party/DiskANN-landmark-global && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/global-landmark-knee/global-build.log
GLOBAL_BIN="$PWD/third_party/DiskANN-landmark-global/target/release/diskann-benchmark"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-landmark-waypoint
git -C third_party/DiskANN-landmark-waypoint checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_waypoint_cache.py third_party/DiskANN-landmark-waypoint
(cd third_party/DiskANN-landmark-waypoint && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/global-landmark-knee/waypoint-build.log
WAYPOINT_BIN="$PWD/third_party/DiskANN-landmark-waypoint/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/landmark-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/global-landmark-knee/results
.venv/bin/python scripts/qualify_global_landmark_knee.py   --global-binary "$GLOBAL_BIN"   --waypoint-binary "$WAYPOINT_BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --landmark-dir "$LANDMARK_DIR"   --router "$ROUTER"   --waypoint "$WAYPOINT"   --work "$WORK"   --out "$PWD/artifacts/global-landmark-knee/results"   --reps 3   2>&1 | tee artifacts/global-landmark-knee/eval.log
