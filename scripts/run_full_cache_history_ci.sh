#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/full-cache-history

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
ROOT="$HOME/.cache/geoivf/full-cache-history-v1"
TRACE_ROOT="$ROOT/traces"
TRACE="$TRACE_ROOT/medoid-teacher.L4.jsonl"
STATE_ROOT="$ROOT/states"

for f in "$BASE" "$QUERIES" "$GT" "$PORTALS"; do test -s "$f"; done
mkdir -p "$TRACE_ROOT" "$STATE_ROOT"
git rev-parse HEAD > artifacts/full-cache-history/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/full-cache-history/host.txt

.venv/bin/python -m py_compile   scripts/collect_medoid_multil_traces.py   scripts/patch_diskann_waypoint_trace.py   scripts/learn_landmarks_trace_prefix.py   scripts/build_hint_ivf.py   scripts/patch_diskann_hint_ivf_packed_direct.py   scripts/qualify_full_cache_history.py

export CARGO_HOME="$RUNNER_TEMP/fullcache-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/fullcache-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/fullcache-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/fullcache-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

# Collect the full 10K ordinary-DiskANN L=4 traversal history once.
if [ ! -s "$TRACE" ]; then
  rm -rf "$TRACE_ROOT"
  mkdir -p "$TRACE_ROOT"
  git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-fullcache-trace
  git -C third_party/DiskANN-fullcache-trace checkout --detach "$DISKANN_REV"
  .venv/bin/python scripts/patch_diskann_waypoint_trace.py third_party/DiskANN-fullcache-trace
  (cd third_party/DiskANN-fullcache-trace && cargo fmt --all &&     cargo build --release --locked -p diskann-benchmark --features disk-index)     2>&1 | tee artifacts/full-cache-history/trace-build.log
  TRACE_BIN="$PWD/third_party/DiskANN-fullcache-trace/target/release/diskann-benchmark"
  WORK="$RUNNER_TEMP/fullcache-trace-$GITHUB_RUN_ID"
  mkdir -p "$WORK"
  .venv/bin/python scripts/collect_medoid_multil_traces.py     --binary "$TRACE_BIN"     --queries "$QUERIES"     --gt "$GT"     --index-prefix "$INDEX_PREFIX"     --work "$WORK"     --out "$TRACE_ROOT"     --train-rows 10000     --teacher-ls "4"     2>&1 | tee artifacts/full-cache-history/trace.log
  rm -rf "$WORK"
fi
test -s "$TRACE"
sha256sum "$TRACE" > artifacts/full-cache-history/full-trace.sha256
cp "$TRACE_ROOT/multi-l-trace-summary.json" artifacts/full-cache-history/

# Build causal prefix states. Rebuild all together if any state is incomplete,
# so all prefixes share one deterministic construction protocol.
missing=0
for h in 5000 6000 7000 8000 9000 10000; do
  test -s "$STATE_ROOT/h$h/landmarks-b16000.bin" || missing=1
  test -s "$STATE_ROOT/h$h/hints-b16000-nlist512-spherical.bin" || missing=1
done
if [ "$missing" = 1 ]; then
  rm -rf "$STATE_ROOT"
  mkdir -p "$STATE_ROOT"
  for h in 5000 6000 7000 8000 9000 10000; do
    out="$STATE_ROOT/h$h"
    mkdir -p "$out"
    .venv/bin/python scripts/learn_landmarks_trace_prefix.py       --trace "$TRACE"       --portals "$PORTALS"       --history-rows "$h"       --budget 16000       --out-dir "$out"       2>&1 | tee "artifacts/full-cache-history/learn-h$h.log"
    .venv/bin/python scripts/build_hint_ivf.py       --base "$BASE"       --hints "$out/landmarks-b16000.bin"       --out "$out/hints-b16000-nlist512-spherical.bin"       --nlist 512       --iterations 5       --batch 2048       --seed 20261005       2>&1 | tee "artifacts/full-cache-history/build-h$h.log"
  done
fi
for h in 5000 6000 7000 8000 9000 10000; do
  cp "$STATE_ROOT/h$h/landmarks.manifest.json"     "artifacts/full-cache-history/h$h-landmarks.manifest.json"
  cp "$STATE_ROOT/h$h/hints-b16000-nlist512-spherical.bin.manifest.json"     "artifacts/full-cache-history/h$h-ivf.manifest.json"
done

# Compile the canonical optimized deployment path once.
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-fullcache-search
git -C third_party/DiskANN-fullcache-search checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py third_party/DiskANN-fullcache-search
(cd third_party/DiskANN-fullcache-search && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/full-cache-history/search-build.log
BIN="$PWD/third_party/DiskANN-fullcache-search/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/fullcache-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/full-cache-history/results
source scripts/perf_guard.sh
geoivf_perf_lock
.venv/bin/python scripts/qualify_full_cache_history.py   --binary "$BIN"   --queries "$QUERIES"   --heldout-gt "$GT"   --index-prefix "$INDEX_PREFIX"   --state-root "$STATE_ROOT"   --work "$WORK"   --out "$PWD/artifacts/full-cache-history/results"   --reps 3   2>&1 | tee artifacts/full-cache-history/eval.log

rm -rf "$WORK"
