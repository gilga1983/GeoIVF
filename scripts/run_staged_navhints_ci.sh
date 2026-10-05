#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/staged-navhints"
ROOT="$HOME/.cache/geoivf/staged-navhints-v1"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART" "$ROOT"

phase() {
  echo
  echo "========== PHASE: $* =========="
  date --iso-8601=seconds
  free -h
}

heartbeat() {
  while true; do
    echo
    echo "========== HEARTBEAT $(date --iso-8601=seconds) =========="
    uptime
    free -h | sed -n '1,2p'
    ps -eo pid,ppid,pcpu,pmem,etime,stat,comm,args --sort=-pcpu | head -n 12
    sleep 60
  done
}
heartbeat &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  wait "$HEARTBEAT_PID" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$PORTALS"; do test -s "$f"; done

phase "validate scripts"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; lsblk -o NAME,TYPE,SIZE,ROTA,MOUNTPOINTS; } > "$ART/host.txt"
.venv/bin/python -m py_compile \
  scripts/learn_global_navigation_landmarks.py \
  scripts/learn_staged_navigation_landmarks.py \
  scripts/build_hint_ivf.py \
  scripts/patch_diskann_staged_navhints.py \
  scripts/qualify_staged_navhints.py

phase "learn canonical 16K and staged 8K+8K vocabularies"
CANON_LAND="$ROOT/canonical-landmarks"
STAGE_LAND="$ROOT/staged-landmarks"
mkdir -p "$CANON_LAND" "$STAGE_LAND"
.venv/bin/python scripts/learn_global_navigation_landmarks.py \
  --trace "$TRACE" --portals "$PORTALS" --out-dir "$CANON_LAND" --budgets "16000" \
  2>&1 | tee "$ART/learn-canonical.log"
.venv/bin/python scripts/learn_staged_navigation_landmarks.py \
  --trace "$TRACE" --out-dir "$STAGE_LAND" --budget 8000 --stages "0,8" \
  2>&1 | tee "$ART/learn-staged.log"
cp "$STAGE_LAND/staged-landmarks.manifest.json" "$ART/"

build_ivf() {
  local hints="$1"
  local out="$2"
  local nlist="$3"
  local log="$4"
  if [ ! -s "$out" ] || [ ! -s "$out.manifest.json" ]; then
    rm -f "$out" "$out.manifest.json"
    .venv/bin/python scripts/build_hint_ivf.py \
      --base "$BASE" --hints "$hints" --out "$out" --nlist "$nlist" \
      --iterations 5 --batch 2048 --seed 20261005 \
      2>&1 | tee "$ART/$log"
  fi
}

CANON_IVF="$ROOT/canonical-b16000-c512.bin"
START_IVF="$ROOT/stage0-b8000-c256.bin"
STAGE_IVF="$ROOT/stage8-b8000-c256.bin"
phase "build ID-only directories"
build_ivf "$CANON_LAND/landmarks-b16000.bin" "$CANON_IVF" 512 "build-canonical.log"
build_ivf "$STAGE_LAND/stage0-b8000.bin" "$START_IVF" 256 "build-stage0.log"
build_ivf "$STAGE_LAND/stage8-b8000.bin" "$STAGE_IVF" 256 "build-stage8.log"
cp "$CANON_IVF.manifest.json" "$ART/canonical.manifest.json"
cp "$START_IVF.manifest.json" "$ART/stage0.manifest.json"
cp "$STAGE_IVF.manifest.json" "$ART/stage8.manifest.json"

phase "build staged DiskANN"
export CARGO_HOME="$RUNNER_TEMP/staged-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/staged-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/staged-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/staged-rustup-$GITHUB_RUN_ID.sh" \
  -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-staged-navhints
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-staged-navhints
git -C third_party/DiskANN-staged-navhints checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_staged_navhints.py third_party/DiskANN-staged-navhints
(cd third_party/DiskANN-staged-navhints && cargo fmt --all)
git -C third_party/DiskANN-staged-navhints diff --check
git -C third_party/DiskANN-staged-navhints diff > "$ART/staged-navhints.patch"
(cd third_party/DiskANN-staged-navhints && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-staged-navhints/target/release/diskann-benchmark"

phase "run equal-memory A/B"
WORK="$RUNNER_TEMP/staged-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_staged_navhints.py \
  --binary "$BIN" \
  --queries "$QUERIES" \
  --gt "$GT" \
  --index-prefix "$INDEX_PREFIX" \
  --ivf-16k "$CANON_IVF" \
  --ivf-start8k "$START_IVF" \
  --ivf-stage8k "$STAGE_IVF" \
  --work "$WORK" \
  --out "$OUT" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
