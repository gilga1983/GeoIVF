#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/landmark-teacher-ablation

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
ROUTER="$HOME/.cache/geoivf/region-ablation-routers-v1/nlist-512/router.bin"
PORTAL_TRACE="$HOME/.cache/geoivf/region-ablation-traces-v1/nlist-512/portal-k1-train5000-trace.jsonl"
MEDOID_TRACE="$HOME/.cache/geoivf/memgraph-medoid-trace-v1/medoid-k1-train5000-trace.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
PORTAL_DIR="$HOME/.cache/geoivf/landmark-teacher-portal-v1"
MEDOID_DIR="$HOME/.cache/geoivf/landmark-teacher-medoid-v1"

for f in "$QUERIES" "$GT" "$ROUTER" "$PORTALS"; do test -s "$f"; done
git rev-parse HEAD > artifacts/landmark-teacher-ablation/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/landmark-teacher-ablation/host.txt

.venv/bin/python -m py_compile   scripts/collect_medoid_navigation_traces.py   scripts/collect_waypoint_traces.py   scripts/learn_global_navigation_landmarks.py   scripts/patch_diskann_global_starts.py   scripts/qualify_landmark_teacher_ablation.py

export CARGO_HOME="$RUNNER_TEMP/teacher-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/teacher-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/teacher-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/teacher-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

# Generate either teacher trace only if the persistent runner cache is missing it.
if [ ! -s "$PORTAL_TRACE" ] || [ ! -s "$MEDOID_TRACE" ]; then
  git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-teacher-trace
  git -C third_party/DiskANN-teacher-trace checkout --detach "$DISKANN_REV"
  .venv/bin/python scripts/patch_diskann_waypoint_trace.py third_party/DiskANN-teacher-trace
  (cd third_party/DiskANN-teacher-trace && cargo fmt --all &&     cargo build --release --locked -p diskann-benchmark --features disk-index)     2>&1 | tee artifacts/landmark-teacher-ablation/trace-build.log
  TRACE_BIN="$PWD/third_party/DiskANN-teacher-trace/target/release/diskann-benchmark"

  if [ ! -s "$MEDOID_TRACE" ]; then
    root="$HOME/.cache/geoivf/memgraph-medoid-trace-v1"
    rm -rf "$root"; mkdir -p "$root"
    work="$RUNNER_TEMP/teacher-medoid-trace-$GITHUB_RUN_ID"
    mkdir -p "$work"
    .venv/bin/python scripts/collect_medoid_navigation_traces.py       --binary "$TRACE_BIN" --queries "$QUERIES" --gt "$GT"       --index-prefix "$INDEX_PREFIX" --work "$work" --out "$root"       --train-rows 5000       2>&1 | tee artifacts/landmark-teacher-ablation/medoid-trace.log
    rm -rf "$work"
  fi

  if [ ! -s "$PORTAL_TRACE" ]; then
    root="$HOME/.cache/geoivf/region-ablation-traces-v1/nlist-512"
    rm -rf "$root"; mkdir -p "$root"
    work="$RUNNER_TEMP/teacher-portal-trace-$GITHUB_RUN_ID"
    mkdir -p "$work"
    .venv/bin/python scripts/collect_waypoint_traces.py       --binary "$TRACE_BIN" --queries "$QUERIES" --gt "$GT"       --index-prefix "$INDEX_PREFIX" --portal-router "$ROUTER"       --work "$work" --out "$root" --train-rows 5000       2>&1 | tee artifacts/landmark-teacher-ablation/portal-trace.log
    rm -rf "$work"
  fi
fi

test -s "$PORTAL_TRACE"
test -s "$MEDOID_TRACE"
sha256sum "$PORTAL_TRACE" "$MEDOID_TRACE" > artifacts/landmark-teacher-ablation/teacher-traces.sha256

rm -rf "$PORTAL_DIR" "$MEDOID_DIR"
mkdir -p "$PORTAL_DIR" "$MEDOID_DIR"
for spec in "portal:$PORTAL_TRACE:$PORTAL_DIR" "medoid:$MEDOID_TRACE:$MEDOID_DIR"; do
  IFS=: read -r name trace out <<< "$spec"
  .venv/bin/python scripts/learn_global_navigation_landmarks.py     --trace "$trace" --portals "$PORTALS" --out-dir "$out"     --budgets "256,512,768,1024,1536,2048,2500"     2>&1 | tee "artifacts/landmark-teacher-ablation/learn-$name.log"
  cp "$out/global-landmarks.manifest.json"     "artifacts/landmark-teacher-ablation/$name-landmarks.manifest.json"
done

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-teacher-global
git -C third_party/DiskANN-teacher-global checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_global_starts.py third_party/DiskANN-teacher-global
(cd third_party/DiskANN-teacher-global && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/landmark-teacher-ablation/global-build.log
GLOBAL_BIN="$PWD/third_party/DiskANN-teacher-global/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/teacher-eval-$GITHUB_RUN_ID"
mkdir -p "$WORK" artifacts/landmark-teacher-ablation/results
.venv/bin/python scripts/qualify_landmark_teacher_ablation.py   --binary "$GLOBAL_BIN"   --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --portal-dir "$PORTAL_DIR" --medoid-dir "$MEDOID_DIR"   --work "$WORK" --out "$PWD/artifacts/landmark-teacher-ablation/results"   --reps 3   2>&1 | tee artifacts/landmark-teacher-ablation/eval.log

rm -rf "$WORK"
