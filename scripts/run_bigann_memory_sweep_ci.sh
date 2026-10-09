#!/usr/bin/env bash
# Three RAM budgets (about 1x,4x,16x) on the frozen BigANN-10M DiskANN graph.
# Self-hosted only, four threads, causal train/warm/eval, genuine NVMe reads.
set -euxo pipefail
DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/bigann-memory-sweep"
WORK="$RUNNER_TEMP/bigann-memory-sweep-$GITHUB_RUN_ID"
BASE_DIR="$HOME/.cache/geoivf/public-eval/bigann-10M"
BASE="$BASE_DIR/data/base.u8bin"
TRAIN="$BASE_DIR/data/train5000.u8bin"
HELD="$BASE_DIR/data/heldout5000.u8bin"
HELD_GT="$BASE_DIR/data/heldout5000.gt"
TRACE="$BASE_DIR/trace/teacher.jsonl"
INDEX_PREFIX="$BASE_DIR/index/diskann-index"
IVF="$BASE_DIR/navhints/hints-b16000-nlist512.bin"
mkdir -p "$ART" "$ART/results" "$ART/manifests" "$WORK"
for f in "$BASE" "$TRAIN" "$HELD" "$HELD_GT" "$TRACE" \
         "$INDEX_PREFIX.complete.json" "$INDEX_PREFIX""_disk.index" \
         "$INDEX_PREFIX""_pq_pivots.bin" "$INDEX_PREFIX""_pq_compressed.bin" \
         "$IVF" "$IVF.manifest.json"; do
  test -s "$f" || { echo "MISSING_FROZEN_PUBLIC_INPUT $f" >&2; exit 4; }
done
git rev-parse HEAD >"$ART/geoivf-commit.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } >"$ART/host.txt"
printf '%s\n' "$DISKANN_REV" >"$ART/diskann-revision.txt"

# Heavy offline fitting and C++/Rust compilation also take the SSD/CPU lock:
# don't perturb the concurrently measured original-system experiment.
LOCK="$HOME/.cache/geoivf/speed-device.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
echo "Waiting for exclusive SSD/CPU preparation lock"
flock 9
echo "Acquired preparation lock"

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6 'faiss-cpu>=1.9,<2'
"$WORK/venv/bin/python" -m py_compile \
  scripts/qualify_bigann_memory_sweep.py \
  scripts/prepare_qsev_bigann.py \
  scripts/patch_diskann_qsev_l2.py

# Exactly the existing static-training and held-out queries; no new vectors.
"$WORK/venv/bin/python" - "$TRAIN" "$HELD" "$HELD_GT" "$WORK" <<'PY'
import struct,sys
from pathlib import Path
train,held,gt,work=map(Path,sys.argv[1:])
def shape(p):
    with p.open("rb") as f: return struct.unpack("<II",f.read(8))
assert shape(train)==shape(held)==(5000,128)
assert shape(gt)==(5000,100)
with (work/"all10000.u8bin").open("wb") as o:
    o.write(struct.pack("<II",10000,128))
    for p in (train,held):
        with p.open("rb") as f:
            f.seek(8);data=f.read()
            assert len(data)==5000*128
            o.write(data)
# Training snapshots skip recall, but benchmark still parses the GT format.
# This dummy is NEVER used for evaluation or presented as ground truth.
with (work/"train-throughput-only.gt").open("wb") as o:
    o.write(struct.pack("<II",5000,100))
    o.write(b"\0"*(5000*100*4*2))
print("BIGANN_QUERY_SPLIT_VERIFIED",flush=True)
PY

HOT="$WORK/hot"
QSEV="$WORK/qsev"
mkdir -p "$HOT" "$QSEV"
"$WORK/venv/bin/python" scripts/learn_hot_cache_ids.py \
  --trace "$TRACE" --out-dir "$HOT" --counts "256,1024,4096" \
  >"$ART/hot-cache-training.log" 2>&1
cp "$HOT/hot-cache.manifest.json" "$ART/manifests/"
"$WORK/venv/bin/python" scripts/prepare_qsev_bigann.py \
  --base "$BASE" --index-prefix "$INDEX_PREFIX" \
  --out-dir "$QSEV" --counts "200,800,3200" \
  --train-size 100000 --seed 12345 --threads 4 \
  >"$ART/qsev-training.log" 2>&1
cp "$QSEV/"*.manifest.json "$ART/manifests/"

export CARGO_HOME="$WORK/cargo"
export RUSTUP_HOME="$WORK/rustup"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-install.sh"
sh "$WORK/rustup-install.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

compile_patch() {
    local kind="$1";shift
    local repo="$WORK/DiskANN-$kind"
    git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$repo"
    git -C "$repo" checkout --detach "$DISKANN_REV"
    for patch in "$@"; do
        "$WORK/venv/bin/python" "scripts/$patch" "$repo"
    done
    (cd "$repo" && cargo fmt --all && git diff --check && \
      cargo build --release --locked -p diskann-benchmark --features disk-index) \
        >"$ART/$kind-build.log" 2>&1
    test -x "$repo/target/release/diskann-benchmark"
}

compile_patch hot patch_diskann_hot_cache.py
compile_patch qsev patch_diskann_qsev_l2.py
compile_patch cat patch_diskann_catapult_snapshot.py
compile_patch nav patch_diskann_hint_ivf_packed_direct.py patch_diskann_experience_core.py
HOT_BIN="$WORK/DiskANN-hot/target/release/diskann-benchmark"
QSEV_BIN="$WORK/DiskANN-qsev/target/release/diskann-benchmark"
CAT_BIN="$WORK/DiskANN-cat/target/release/diskann-benchmark"
NAV_BIN="$WORK/DiskANN-nav/target/release/diskann-benchmark"

# Release preparation lock; qualifier acquires the same lock for the entire
# measurement epoch and rotates arms inside it. Do not nest flock handles.
flock -u 9
exec 9>&-

"$WORK/venv/bin/python" scripts/qualify_bigann_memory_sweep.py \
  --hot-binary "$HOT_BIN" --qsev-binary "$QSEV_BIN" \
  --cat-binary "$CAT_BIN" --nav-binary "$NAV_BIN" \
  --queries "$WORK/all10000.u8bin" --heldout-gt "$HELD_GT" \
  --train-gt "$WORK/train-throughput-only.gt" \
  --index-prefix "$INDEX_PREFIX" --ivf "$IVF" \
  --hot-dir "$HOT" --qsev-dir "$QSEV" \
  --work "$WORK/eval" --out "$ART/results" --reps 3 \
  >"$ART/eval.log" 2>&1
test -s "$ART/results/bigann-memory-sweep.json"
echo "BIGANN_MEMORY_SWEEP_SUCCESS"
