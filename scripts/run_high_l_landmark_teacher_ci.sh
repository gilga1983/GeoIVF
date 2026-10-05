#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
TEACHER_LS="1 4 8 16 32 64"
BUDGETS="1024,1536,2048"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p artifacts/high-l-landmark-teacher

QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
PORTAL_TRACE="$HOME/.cache/geoivf/region-ablation-traces-v1/nlist-512/portal-k1-train5000-trace.jsonl"
TRACE_ROOT="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1"
TEACHERS_ROOT="$HOME/.cache/geoivf/high-l-landmark-teachers-v1"
PORTAL_DIR="$HOME/.cache/geoivf/high-l-landmark-portal-reference-v1"

for f in "$QUERIES" "$GT" "$PORTALS" "$PORTAL_TRACE"; do test -s "$f"; done
mkdir -p "$TRACE_ROOT" "$TEACHERS_ROOT" "$PORTAL_DIR"
git rev-parse HEAD > artifacts/high-l-landmark-teacher/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/high-l-landmark-teacher/host.txt

.venv/bin/python -m py_compile   scripts/collect_medoid_multil_traces.py   scripts/learn_global_navigation_landmarks.py   scripts/patch_diskann_waypoint_trace.py   scripts/patch_diskann_global_starts.py   scripts/qualify_high_l_landmark_teacher.py

export CARGO_HOME="$RUNNER_TEMP/highl-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/highl-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/highl-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/highl-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

missing=0
for l in $TEACHER_LS; do
  [ -s "$TRACE_ROOT/medoid-teacher.L${l}.jsonl" ] || missing=1
done

if [ "$missing" = 1 ]; then
  rm -rf "$TRACE_ROOT"
  mkdir -p "$TRACE_ROOT"
  git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-highl-trace
  git -C third_party/DiskANN-highl-trace checkout --detach "$DISKANN_REV"
  .venv/bin/python scripts/patch_diskann_waypoint_trace.py third_party/DiskANN-highl-trace
  (cd third_party/DiskANN-highl-trace && cargo fmt --all &&     cargo build --release --locked -p diskann-benchmark --features disk-index)     2>&1 | tee artifacts/high-l-landmark-teacher/trace-build.log
  TRACE_BIN="$PWD/third_party/DiskANN-highl-trace/target/release/diskann-benchmark"

  work="$RUNNER_TEMP/highl-trace-$GITHUB_RUN_ID"
  mkdir -p "$work"
  .venv/bin/python scripts/collect_medoid_multil_traces.py     --binary "$TRACE_BIN"     --queries "$QUERIES"     --gt "$GT"     --index-prefix "$INDEX_PREFIX"     --work "$work"     --out "$TRACE_ROOT"     --train-rows 5000     --teacher-ls "1,4,8,16,32,64"     2>&1 | tee artifacts/high-l-landmark-teacher/trace.log
  rm -rf "$work"
fi

cp "$TRACE_ROOT/multi-l-trace-summary.json" artifacts/high-l-landmark-teacher/
sha256sum "$TRACE_ROOT"/medoid-teacher.L*.jsonl > artifacts/high-l-landmark-teacher/teacher-traces.sha256

rm -rf "$TEACHERS_ROOT"
mkdir -p "$TEACHERS_ROOT"
for l in $TEACHER_LS; do
  out="$TEACHERS_ROOT/L$l"
  mkdir -p "$out"
  .venv/bin/python scripts/learn_global_navigation_landmarks.py     --trace "$TRACE_ROOT/medoid-teacher.L${l}.jsonl"     --portals "$PORTALS"     --out-dir "$out"     --budgets "$BUDGETS"     2>&1 | tee "artifacts/high-l-landmark-teacher/learn-L${l}.log"
  cp "$out/global-landmarks.manifest.json"     "artifacts/high-l-landmark-teacher/medoid-L${l}.manifest.json"
done

rm -rf "$PORTAL_DIR"
mkdir -p "$PORTAL_DIR"
.venv/bin/python scripts/learn_global_navigation_landmarks.py   --trace "$PORTAL_TRACE"   --portals "$PORTALS"   --out-dir "$PORTAL_DIR"   --budgets "$BUDGETS"   2>&1 | tee artifacts/high-l-landmark-teacher/learn-portal.log
cp "$PORTAL_DIR/global-landmarks.manifest.json"   artifacts/high-l-landmark-teacher/portal.manifest.json

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-highl-global
git -C third_party/DiskANN-highl-global checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_global_starts.py third_party/DiskANN-highl-global
(cd third_party/DiskANN-highl-global && cargo fmt --all &&   cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/high-l-landmark-teacher/global-build.log
GLOBAL_BIN="$PWD/third_party/DiskANN-highl-global/target/release/diskann-benchmark"

work="$RUNNER_TEMP/highl-eval-$GITHUB_RUN_ID"
mkdir -p "$work" artifacts/high-l-landmark-teacher/results
.venv/bin/python scripts/qualify_high_l_landmark_teacher.py   --binary "$GLOBAL_BIN"   --queries "$QUERIES"   --gt "$GT"   --index-prefix "$INDEX_PREFIX"   --teachers-root "$TEACHERS_ROOT"   --portal-dir "$PORTAL_DIR"   --work "$work"   --out "$PWD/artifacts/high-l-landmark-teacher/results"   --reps 3   2>&1 | tee artifacts/high-l-landmark-teacher/eval.log
rm -rf "$work"
