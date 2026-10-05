#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/hint-ivf-profile

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/hint-ivf-profile-v1"
LANDMARK_DIR="$ROOT/landmarks"
IVF="$ROOT/hints-b16000-nlist512-spherical.bin"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done
rm -rf "$ROOT"
mkdir -p "$LANDMARK_DIR"
git rev-parse HEAD > artifacts/hint-ivf-profile/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/hint-ivf-profile/host.txt

.venv/bin/python -m py_compile   scripts/learn_global_navigation_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_hint_ivf_profile.py   scripts/qualify_hint_ivf_profile.py

.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$TRACE"   --portals "$PORTALS"   --out-dir "$LANDMARK_DIR"   --budgets "16000"   2>&1 | tee artifacts/hint-ivf-profile/learn.log

.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE"   --hints "$LANDMARK_DIR/landmarks-b16000.bin"   --out "$IVF"   --nlist 512   --iterations 5   --batch 2048   --seed 20261005   2>&1 | tee artifacts/hint-ivf-profile/build-ivf.log
cp "$IVF.manifest.json" artifacts/hint-ivf-profile/

export CARGO_HOME="$RUNNER_TEMP/hintprofile-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/hintprofile-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/hintprofile-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/hintprofile-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-hint-profile
git -C third_party/DiskANN-hint-profile checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_profile.py third_party/DiskANN-hint-profile
(
  cd third_party/DiskANN-hint-profile
  cargo fmt --all
  cargo fmt --all -- --check
  cargo check --release --locked -p diskann-benchmark --features disk-index
  cargo build --release --locked -p diskann-benchmark --features disk-index
) 2>&1 | tee artifacts/hint-ivf-profile/build.log
BIN="$PWD/third_party/DiskANN-hint-profile/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/hintprofile-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/hint-ivf-profile/results
.venv/bin/python scripts/qualify_hint_ivf_profile.py   --binary "$BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --ivf "$IVF"   --work "$WORK"   --out "$PWD/artifacts/hint-ivf-profile/results"   --reps 3   2>&1 | tee artifacts/hint-ivf-profile/eval.log

rm -rf "$WORK"
