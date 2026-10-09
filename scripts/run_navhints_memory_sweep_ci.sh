#!/usr/bin/env bash
# Self-hosted only. Same pinned DiskANN and frozen PubMed/MedCPT benchmark
# across increasing cache, QSEV and Catapult memory budgets.
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/navhints-memory-sweep"
WORK="$RUNNER_TEMP/navhints-memory-sweep-$GITHUB_RUN_ID"
mkdir -p "$ART" "$WORK" "$ART/results" "$ART/manifests"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6 "faiss-cpu>=1.9,<2"

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
HELD_GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
TRAIN_GT="$HOME/.cache/geoivf/catapult-paper-pubmed1m/throughput-only.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
IVF="$HOME/.cache/geoivf/paper-competitors-phase1-v1/navhints/hints-b16000-nlist512-spherical.bin"
for f in "$QUERIES" "$HELD_GT" "$TRAIN_GT" "$BASE" "$TRACE" "$IVF" \
         "$IVF.manifest.json" \
         "$INDEX_PREFIX""_disk.index" "$INDEX_PREFIX""_pq_pivots.bin" \
         "$INDEX_PREFIX""_pq_compressed.bin"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
printf 'Pinned Microsoft DiskANN revision: %s\n' "$DISKANN_REV" > "$ART/diskann-revision.txt"
.venv/bin/python -m py_compile \
    scripts/patch_diskann_catapult_snapshot.py \
    scripts/patch_diskann_hot_cache.py \
    scripts/patch_diskann_qsev.py \
    scripts/patch_diskann_hint_ivf_packed_direct.py \
    scripts/patch_diskann_experience_core.py \
    scripts/qualify_final_catapult.py \
    scripts/qualify_navhints_memory_sweep.py \
    scripts/prepare_diskannpp_qsev.py

HOT="$WORK/hot"
QSEV="$WORK/qsev"
mkdir -p "$HOT" "$QSEV"
.venv/bin/python scripts/learn_hot_cache_ids.py \
    --trace "$TRACE" --out-dir "$HOT" --counts "30,60,120,240,480" \
    2>&1 | tee "$ART/hot-cache-build.log"
cp "$HOT/hot-cache.manifest.json" "$ART/manifests/"

for n in 32 64 128 256 512; do
    c=$((n-1))
    out="$QSEV/qsev-$n.bin"
    .venv/bin/python scripts/prepare_diskannpp_qsev.py \
      --base "$BASE" --index-prefix "$INDEX_PREFIX" \
      --out "$out" --clusters "$c" \
      --train-size 100000 --seed 12345 --threads 4 \
      2>&1 | tee "$ART/qsev-$n-train.log"
    cp "$QSEV/qsev-$n.manifest.json" "$ART/manifests/qsev-$n.manifest.json"
done

export CARGO_HOME="$WORK/cargo"
export RUSTUP_HOME="$WORK/rustup"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-install.sh"
sh "$WORK/rustup-install.sh" -y --profile minimal \
  --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

compile_patch() {
    local kind="$1"; shift
    local checkout="$WORK/DiskANN-$kind"
    git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$checkout"
    git -C "$checkout" checkout --detach "$DISKANN_REV"
    test "$(git -C "$checkout" rev-parse HEAD)" = "$DISKANN_REV"
    local patch
    for patch in "$@"; do
        .venv/bin/python "scripts/$patch" "$checkout"
    done
    (cd "$checkout" && cargo fmt --all && git diff --check && \
     cargo build --release --locked -p diskann-benchmark --features disk-index) \
       2>&1 | tee "$ART/$kind-build.log"
}

compile_patch hot patch_diskann_hot_cache.py
compile_patch qsev patch_diskann_qsev.py
compile_patch cat patch_diskann_catapult_snapshot.py
compile_patch nav patch_diskann_hint_ivf_packed_direct.py patch_diskann_experience_core.py

HOT_BIN="$WORK/DiskANN-hot/target/release/diskann-benchmark"
QSEV_BIN="$WORK/DiskANN-qsev/target/release/diskann-benchmark"
CAT_BIN="$WORK/DiskANN-cat/target/release/diskann-benchmark"
NAV_BIN="$WORK/DiskANN-nav/target/release/diskann-benchmark"
for f in "$HOT_BIN" "$QSEV_BIN" "$CAT_BIN" "$NAV_BIN"; do test -x "$f"; done

.venv/bin/python scripts/qualify_navhints_memory_sweep.py \
  --hot-binary "$HOT_BIN" \
  --qsev-binary "$QSEV_BIN" \
  --cat-binary "$CAT_BIN" \
  --nav-binary "$NAV_BIN" \
  --queries "$QUERIES" \
  --heldout-gt "$HELD_GT" \
  --train-gt "$TRAIN_GT" \
  --index-prefix "$INDEX_PREFIX" \
  --ivf "$IVF" \
  --hot-dir "$HOT" \
  --qsev-dir "$QSEV" \
  --work "$WORK/evaluation" \
  --out "$ART/results" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"

printf 'All budgets, cache-hit verification and matched-recall results complete.\n'
rm -rf "$WORK"
