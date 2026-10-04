#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6 "faiss-cpu>=1.9,<2"
mkdir -p artifacts/id-portal-sweep

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
ROUTER="$HOME/.cache/geoivf/region-ablation-routers-v1/nlist-512/router.bin"
WAYPOINT="$HOME/.cache/geoivf/region-ablation-caches-v1/nlist-512/heavy-skip-b2500.bin"
POOL_DIR="$HOME/.cache/geoivf/id-portal-pools-v1"

for f in "$BASE" "$QUERIES" "$GT" "$ROUTER" "$WAYPOINT"; do test -s "$f"; done
mkdir -p "$POOL_DIR"
git rev-parse HEAD > artifacts/id-portal-sweep/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/id-portal-sweep/host.txt

.venv/bin/python -m py_compile   scripts/prepare_id_portal_pools.py   scripts/patch_diskann_global_starts.py   scripts/qualify_id_portal_sweep.py

if [ ! -s "$POOL_DIR/id-portal-pools.manifest.json" ]; then
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"
  .venv/bin/python scripts/prepare_id_portal_pools.py     --base "$BASE"     --router "$ROUTER"     --waypoint-cache "$WAYPOINT"     --out-dir "$POOL_DIR"     --per-cell-reservoir 256     --max-portals-per-region 64     --counts "512,1024,2048,4096,8192,16384"     --seed 12345     --threads 4     --chunk 32768     2>&1 | tee artifacts/id-portal-sweep/pool-build.log
fi
cp "$POOL_DIR/id-portal-pools.manifest.json" artifacts/id-portal-sweep/pool-manifest.json

export CARGO_HOME="$RUNNER_TEMP/idportal-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/idportal-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/idportal-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/idportal-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-idportal-global
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-idportal-global
git -C third_party/DiskANN-idportal-global checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_global_starts.py third_party/DiskANN-idportal-global
(cd third_party/DiskANN-idportal-global && cargo fmt --all)
git -C third_party/DiskANN-idportal-global diff --check
git -C third_party/DiskANN-idportal-global diff > artifacts/id-portal-sweep/global-starts.patch
(cd third_party/DiskANN-idportal-global &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/id-portal-sweep/global-build.log
GLOBAL_BIN="$PWD/third_party/DiskANN-idportal-global/target/release/diskann-benchmark"

rm -rf third_party/DiskANN-idportal-waypoint
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-idportal-waypoint
git -C third_party/DiskANN-idportal-waypoint checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_waypoint_cache.py third_party/DiskANN-idportal-waypoint
(cd third_party/DiskANN-idportal-waypoint && cargo fmt --all)
git -C third_party/DiskANN-idportal-waypoint diff --check
(cd third_party/DiskANN-idportal-waypoint &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/id-portal-sweep/waypoint-build.log
WAYPOINT_BIN="$PWD/third_party/DiskANN-idportal-waypoint/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/idportal-eval-$GITHUB_RUN_ID"
rm -rf "$WORK"
mkdir -p "$WORK" artifacts/id-portal-sweep/results

.venv/bin/python scripts/qualify_id_portal_sweep.py   --global-binary "$GLOBAL_BIN"   --waypoint-binary "$WAYPOINT_BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --pool-dir "$POOL_DIR"   --router "$ROUTER"   --waypoint "$WAYPOINT"   --work "$WORK"   --out "$PWD/artifacts/id-portal-sweep/results"   --train-rows 5000   --reps 3   2>&1 | tee artifacts/id-portal-sweep/eval.log

rm -rf "$WORK"
