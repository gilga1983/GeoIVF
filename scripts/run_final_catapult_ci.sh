#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/final-catapult"
mkdir -p "$ART"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
HELD_GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
TRAIN_GT="$HOME/.cache/geoivf/catapult-paper-pubmed1m/throughput-only.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
IVF16="$HOME/.cache/geoivf/paper-competitors-phase1-v1/navhints/hints-b16000-nlist512-spherical.bin"
for f in "$QUERIES" "$HELD_GT" "$TRAIN_GT" "$IVF16" "$INDEX_PREFIX"_disk.index "$INDEX_PREFIX"_pq_pivots.bin "$INDEX_PREFIX"_pq_compressed.bin; do
  test -s "$f"
done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
.venv/bin/python -m py_compile   scripts/patch_diskann_catapult_snapshot.py   scripts/patch_diskann_hint_ivf_packed_direct.py   scripts/patch_diskann_experience_core.py   scripts/qualify_final_catapult.py

export CARGO_HOME="$RUNNER_TEMP/finalcat-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/finalcat-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/finalcat-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/finalcat-rustup-$GITHUB_RUN_ID.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

WORK="$RUNNER_TEMP/final-catapult-$GITHUB_RUN_ID"
rm -rf "$WORK"; mkdir -p "$WORK"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-cat"
git -C "$WORK/DiskANN-cat" checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_catapult_snapshot.py "$WORK/DiskANN-cat"
(cd "$WORK/DiskANN-cat" && cargo fmt --all && git diff --check && cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/catapult-build.log"
CAT_BIN="$WORK/DiskANN-cat/target/release/diskann-benchmark"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-nav"
git -C "$WORK/DiskANN-nav" checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/DiskANN-nav"
.venv/bin/python scripts/patch_diskann_experience_core.py "$WORK/DiskANN-nav"
(cd "$WORK/DiskANN-nav" && cargo fmt --all && git diff --check && cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/navhints-build.log"
NAV_BIN="$WORK/DiskANN-nav/target/release/diskann-benchmark"

OUT="$ART/results"; mkdir -p "$OUT"
.venv/bin/python scripts/qualify_final_catapult.py   --catapult-binary "$CAT_BIN"   --nav-binary "$NAV_BIN"   --queries "$QUERIES"   --heldout-gt "$HELD_GT"   --training-gt "$TRAIN_GT"   --index-prefix "$INDEX_PREFIX"   --ivf-16k "$IVF16"   --work "$WORK/eval"   --out "$OUT"   --reps 3   2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
