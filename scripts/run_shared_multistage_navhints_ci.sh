#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/shared-multistage-navhints"
ROOT="$HOME/.cache/geoivf/shared-multistage-navhints-v1"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART" "$ROOT"

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/onpolicy-trace/navhints8k-teacher-L4.jsonl"
STAGE0_IDS="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/offpolicy-landmarks/stage0-b8000.bin"
CANON_IVF="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/canonical-b16000-c512.bin"
START_IVF="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/stage0-b8000-c256.bin"
STAGE8K_IVF="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1/onpolicy-stage3-b8000-c256.bin"
for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$STAGE0_IDS" "$CANON_IVF" "$START_IVF" "$STAGE8K_IVF"; do
  test -s "$f"
done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/learn_disjoint_multistage_landmarks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_shared_multistage_navhints.py   scripts/qualify_staged_navhints.py   scripts/qualify_shared_multistage_navhints.py

LAND="$ROOT/residual-landmarks"
rm -rf "$LAND"; mkdir -p "$LAND"
.venv/bin/python scripts/learn_disjoint_multistage_landmarks.py   --trace "$TRACE" --stage0 "$STAGE0_IDS"   --stages "3" --budgets "4096" --out-dir "$LAND"   2>&1 | tee "$ART/learn-residual4k.log"
cp "$LAND/manifest.json" "$ART/residual4k.manifest.json"

RESIDUAL_IVF="$ROOT/residual-stage3-b4096-c128.bin"
rm -f "$RESIDUAL_IVF" "$RESIDUAL_IVF.manifest.json"
.venv/bin/python scripts/build_hint_ivf.py   --base "$BASE" --hints "$LAND/stage3-b4096.bin"   --out "$RESIDUAL_IVF" --nlist 128   --iterations 5 --batch 2048 --seed 20261005   2>&1 | tee "$ART/build-residual4k.log"
cp "$RESIDUAL_IVF.manifest.json" "$ART/residual4k-ivf.manifest.json"

export CARGO_HOME="$RUNNER_TEMP/sharedmulti-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/sharedmulti-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/sharedmulti-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/sharedmulti-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-shared-multistage
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-shared-multistage
git -C third_party/DiskANN-shared-multistage checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_shared_multistage_navhints.py third_party/DiskANN-shared-multistage
(cd third_party/DiskANN-shared-multistage && cargo fmt --all)
git -C third_party/DiskANN-shared-multistage diff --check
git -C third_party/DiskANN-shared-multistage diff > "$ART/shared-multistage.patch"
(cd third_party/DiskANN-shared-multistage && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-shared-multistage/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/sharedmulti-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_shared_multistage_navhints.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --ivf-16k "$CANON_IVF" --ivf-start8k "$START_IVF"   --ivf-stage8k-onpolicy "$STAGE8K_IVF" --ivf-residual4k "$RESIDUAL_IVF"   --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"
rm -rf "$WORK"
