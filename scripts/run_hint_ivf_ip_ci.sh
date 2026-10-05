#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-ivf-ip

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-ivf-ip-v1"
LANDMARK_DIR="$ROOT/landmarks"
IP_ROOT="$ROOT/ip"
SPHERICAL="$ROOT/hints-b16000-nlist256-spherical.bin"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_DIR" "$IP_ROOT"
git rev-parse HEAD > artifacts/hint-ivf-ip/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-ivf-ip/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/build_hint_ivf_ip.py   scripts/patch_diskann_hint_ivf.py   scripts/qualify_hint_ivf_ip_sweep.py

.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "2048,16000"   2>&1 | tee artifacts/hint-ivf-ip/learn.log
cp "$LANDMARK_DIR/global-landmarks.manifest.json" artifacts/hint-ivf-ip/

.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE"   --hints "$LANDMARK_DIR/landmarks-b16000.bin"   --out "$SPHERICAL"   --nlist 256   --iterations 5   --batch 2048   2>&1 | tee artifacts/hint-ivf-ip/build-spherical256.log
cp "$SPHERICAL.manifest.json" artifacts/hint-ivf-ip/

for nlist in 128 256 512; do
  out="$IP_ROOT/hints-b16000-nlist${nlist}-ip.bin"
  .venv/bin/python scripts/build_hint_ivf_ip.py     --base "$BASE"     --hints "$LANDMARK_DIR/landmarks-b16000.bin"     --out "$out"     --nlist "$nlist"     --iterations 4     --batch 1024     --seed 20261005     2>&1 | tee "artifacts/hint-ivf-ip/build-ip${nlist}.log"
  cp "$out.manifest.json" "artifacts/hint-ivf-ip/ip${nlist}.manifest.json"
done

export CARGO_HOME="$RUNNER_TEMP/hintivfip-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintivfip-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintivfip-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintivfip-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hint-ivf-ip
git -C third_party/DiskANN-hint-ivf-ip checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf.py third_party/DiskANN-hint-ivf-ip
(cd third_party/DiskANN-hint-ivf-ip && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/hint-ivf-ip/build.log
BIN="$PWD/third_party/DiskANN-hint-ivf-ip/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintivfip-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-ivf-ip/results
.venv/bin/python scripts/qualify_hint_ivf_ip_sweep.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --flat-2048 "$LANDMARK_DIR/landmarks-b2048.bin"   --flat-16000 "$LANDMARK_DIR/landmarks-b16000.bin"   --ip-root "$IP_ROOT"   --spherical-256 "$SPHERICAL"   --work "$WORK"   --out "$PWD/artifacts/hint-ivf-ip/results"   --reps 3   2>&1 | tee artifacts/hint-ivf-ip/eval.log

rm -rf "$WORK"
