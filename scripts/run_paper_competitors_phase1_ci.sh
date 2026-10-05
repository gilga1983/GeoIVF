#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/paper-competitors-phase1"
ROOT="$HOME/.cache/geoivf/paper-competitors-phase1-v1"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6 "faiss-cpu>=1.9,<2"
mkdir -p "$ART" "$ROOT"

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"

for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do
  test -s "$f"
done

# The public 10M suite can use ~32 GiB while building its DiskANN index.
# All self-hosted runners share the same laptop, so do not overlap our FAISS
# QSEV build with that memory-heavy phase. The suite parent process stays
# alive across both public datasets, making this a simple machine-local gate.
while ps -eo args= | grep -Eq '[r]un_public_dataset_suite_ci\.sh|[r]un_public_dataset_one\.sh|[b]uild_public_diskann_index\.py'; do
  echo "public 10M suite still active; deferring competitor state build"
  sleep 60
done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; lsblk -o NAME,TYPE,SIZE,ROTA,MOUNTPOINTS; } > "$ART/host.txt"

.venv/bin/python -m py_compile \
  scripts/learn_hot_cache_ids.py \
  scripts/prepare_diskannpp_qsev.py \
  scripts/patch_diskann_hot_cache.py \
  scripts/patch_diskann_qsev.py \
  scripts/patch_diskann_hint_ivf_packed_direct.py \
  scripts/learn_global_navigation_landmarks.py \
  scripts/build_hint_ivf.py \
  scripts/qualify_paper_competitors_phase1.py

HELDOUT="$RUNNER_TEMP/paper-comp-heldout-$GITHUB_RUN_ID.fbin"
.venv/bin/python - "$QUERIES" "$HELDOUT" <<'PY'
import struct
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
with src.open("rb") as f:
    rows, dim = struct.unpack("<II", f.read(8))
    assert (rows, dim) == (10000, 768)
    f.seek(8 + 5000 * dim * 4)
    body = f.read(5000 * dim * 4)
    assert len(body) == 5000 * dim * 4
with dst.open("wb") as f:
    f.write(struct.pack("<II", 5000, dim))
    f.write(body)
PY

HOT_DIR="$ROOT/hot"
mkdir -p "$HOT_DIR"
.venv/bin/python scripts/learn_hot_cache_ids.py \
  --trace "$TRACE" \
  --out-dir "$HOT_DIR" \
  --counts "30,32" \
  2>&1 | tee "$ART/hot-cache-build.log"
cp "$HOT_DIR/hot-cache.manifest.json" "$ART/"

QSEV_DIR="$ROOT/qsev"
QSEV="$QSEV_DIR/qsev-32.bin"
mkdir -p "$QSEV_DIR"
if [ ! -s "$QSEV" ] || [ ! -s "$QSEV_DIR/qsev-32.manifest.json" ]; then
  rm -f "$QSEV" "$QSEV_DIR/qsev-32.manifest.json"
  .venv/bin/python scripts/prepare_diskannpp_qsev.py \
    --base "$BASE" \
    --index-prefix "$INDEX_PREFIX" \
    --out "$QSEV" \
    --clusters 31 \
    --train-size 100000 \
    --seed 12345 \
    --threads 4 \
    2>&1 | tee "$ART/qsev-build.log"
fi
test -s "$QSEV"
test -s "$QSEV_DIR/qsev-32.manifest.json"
cp "$QSEV_DIR/qsev-32.manifest.json" "$ART/"
sha256sum "$QSEV" > "$ART/qsev-32.sha256"

NAV_DIR="$ROOT/navhints"
LANDMARK_DIR="$NAV_DIR/landmarks"
IVF="$NAV_DIR/hints-b16000-nlist512-spherical.bin"
mkdir -p "$LANDMARK_DIR"
if [ ! -s "$LANDMARK_DIR/landmarks-b16000.bin" ]; then
  .venv/bin/python scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" \
    --portals "$PORTALS" \
    --out-dir "$LANDMARK_DIR" \
    --budgets "16000" \
    2>&1 | tee "$ART/navhints-learn.log"
fi
if [ ! -s "$IVF" ] || [ ! -s "$IVF.manifest.json" ]; then
  rm -f "$IVF" "$IVF.manifest.json"
  .venv/bin/python scripts/build_hint_ivf.py \
    --base "$BASE" \
    --hints "$LANDMARK_DIR/landmarks-b16000.bin" \
    --out "$IVF" \
    --nlist 512 \
    --iterations 5 \
    --batch 2048 \
    --seed 20261005 \
    2>&1 | tee "$ART/navhints-ivf-build.log"
fi
test -s "$IVF"
test -s "$IVF.manifest.json"
cp "$IVF.manifest.json" "$ART/"
sha256sum "$IVF" > "$ART/navhints-ivf.sha256"

export CARGO_HOME="$RUNNER_TEMP/papercomp-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/papercomp-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/papercomp-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/papercomp-rustup-$GITHUB_RUN_ID.sh" \
  -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-paper-hot third_party/DiskANN-paper-qsev third_party/DiskANN-paper-nav

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-paper-hot
git -C third_party/DiskANN-paper-hot checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hot_cache.py third_party/DiskANN-paper-hot
(cd third_party/DiskANN-paper-hot && cargo fmt --all)
git -C third_party/DiskANN-paper-hot diff --check
git -C third_party/DiskANN-paper-hot diff > "$ART/hot-cache.patch"
(cd third_party/DiskANN-paper-hot && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/hot-build.log"
HOT_BIN="$PWD/third_party/DiskANN-paper-hot/target/release/diskann-benchmark"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-paper-qsev
git -C third_party/DiskANN-paper-qsev checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_qsev.py third_party/DiskANN-paper-qsev
(cd third_party/DiskANN-paper-qsev && cargo fmt --all)
git -C third_party/DiskANN-paper-qsev diff --check
git -C third_party/DiskANN-paper-qsev diff > "$ART/qsev.patch"
(cd third_party/DiskANN-paper-qsev && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/qsev-build-rust.log"
QSEV_BIN="$PWD/third_party/DiskANN-paper-qsev/target/release/diskann-benchmark"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-paper-nav
git -C third_party/DiskANN-paper-nav checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py third_party/DiskANN-paper-nav
(cd third_party/DiskANN-paper-nav && cargo fmt --all)
git -C third_party/DiskANN-paper-nav diff --check
git -C third_party/DiskANN-paper-nav diff > "$ART/navhints.patch"
(cd third_party/DiskANN-paper-nav && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/navhints-build.log"
NAV_BIN="$PWD/third_party/DiskANN-paper-nav/target/release/diskann-benchmark"

OUT="$ART/results"
mkdir -p "$OUT"
.venv/bin/python scripts/qualify_paper_competitors_phase1.py \
  --hot-binary "$HOT_BIN" \
  --qsev-binary "$QSEV_BIN" \
  --nav-binary "$NAV_BIN" \
  --queries "$HELDOUT" \
  --gt "$GT" \
  --index-prefix "$INDEX_PREFIX" \
  --hot-dir "$HOT_DIR" \
  --qsev "$QSEV" \
  --ivf "$IVF" \
  --out "$OUT" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"

rm -f "$HELDOUT"
