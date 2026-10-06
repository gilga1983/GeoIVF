#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/paper-memory-frontier"
ROOT="$HOME/.cache/geoivf/paper-memory-frontier-v1"

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

phase "bootstrap and validate"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; lsblk -o NAME,TYPE,SIZE,ROTA,MOUNTPOINTS; } > "$ART/host.txt"
.venv/bin/python -m py_compile \
  scripts/learn_global_navigation_landmarks.py \
  scripts/build_hint_ivf.py \
  scripts/patch_diskann_hint_ivf_packed_direct.py \
  scripts/patch_diskann_progressive_navhints.py \
  scripts/qualify_paper_memory_frontier.py

LANDMARK_DIR="$ROOT/landmarks"
mkdir -p "$LANDMARK_DIR"
phase "learn 2K 4K 8K 16K landmark vocabularies"
.venv/bin/python scripts/learn_global_navigation_landmarks.py \
  --trace "$TRACE" \
  --portals "$PORTALS" \
  --out-dir "$LANDMARK_DIR" \
  --budgets "2048,4096,8192,16000" \
  2>&1 | tee "$ART/learn.log"
cp "$LANDMARK_DIR/global-landmarks.manifest.json" "$ART/"

build_state() {
  local budget="$1"
  local nlist="$2"
  local out="$ROOT/hints-b${budget}-nlist${nlist}-spherical.bin"
  phase "build state B=${budget} C=${nlist}"
  if [ ! -s "$out" ] || [ ! -s "$out.manifest.json" ]; then
    rm -f "$out" "$out.manifest.json"
    .venv/bin/python scripts/build_hint_ivf.py \
      --base "$BASE" \
      --hints "$LANDMARK_DIR/landmarks-b${budget}.bin" \
      --out "$out" \
      --nlist "$nlist" \
      --iterations 5 \
      --batch 2048 \
      --seed 20261005 \
      2>&1 | tee "$ART/build-b${budget}.log"
  fi
  cp "$out.manifest.json" "$ART/ivf-b${budget}.manifest.json"
  sha256sum "$out" > "$ART/ivf-b${budget}.sha256"
}

build_state 2048 64
build_state 4096 128
build_state 8192 256
build_state 16000 512

phase "build optimized DiskANN"
export CARGO_HOME="$RUNNER_TEMP/memfront-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/memfront-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/memfront-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/memfront-rustup-$GITHUB_RUN_ID.sh" \
  -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-paper-memory
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-paper-memory
git -C third_party/DiskANN-paper-memory checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_progressive_navhints.py third_party/DiskANN-paper-memory
(cd third_party/DiskANN-paper-memory && cargo fmt --all)
git -C third_party/DiskANN-paper-memory diff --check
git -C third_party/DiskANN-paper-memory diff > "$ART/navhints-progressive.patch"
(cd third_party/DiskANN-paper-memory && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-paper-memory/target/release/diskann-benchmark"

phase "run memory frontier"
WORK="$RUNNER_TEMP/memfront-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_paper_memory_frontier.py \
  --binary "$BIN" \
  --queries "$QUERIES" \
  --gt "$GT" \
  --index-prefix "$INDEX_PREFIX" \
  --ivf-2048 "$ROOT/hints-b2048-nlist64-spherical.bin" \
  --ivf-4096 "$ROOT/hints-b4096-nlist128-spherical.bin" \
  --ivf-8192 "$ROOT/hints-b8192-nlist256-spherical.bin" \
  --ivf-16000 "$ROOT/hints-b16000-nlist512-spherical.bin" \
  --work "$WORK" \
  --out "$OUT" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
