#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/onpolicy-staged-navhints"
ROOT="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1"
STAGE=3

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
MEDOID_TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$HOME/.cache/geoivf/id-portal-pools-v1/portals-n512.bin"
for f in "$BASE" "$QUERIES" "$GT" "$MEDOID_TRACE" "$PORTALS"; do test -s "$f"; done

phase "validate sources"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; lsblk -o NAME,TYPE,SIZE,ROTA,MOUNTPOINTS; } > "$ART/host.txt"
.venv/bin/python -m py_compile \
  scripts/learn_global_navigation_landmarks.py \
  scripts/learn_staged_navigation_landmarks.py \
  scripts/build_hint_ivf.py \
  scripts/patch_diskann_navhints_trace.py \
  scripts/collect_navhints_onpolicy_traces.py \
  scripts/patch_diskann_staged_navhints.py \
  scripts/qualify_staged_navhints.py \
  scripts/qualify_onpolicy_staged_navhints.py

CANON_LAND="$ROOT/canonical-landmarks"
OFF_LAND="$ROOT/offpolicy-landmarks"
ON_TRACE_DIR="$ROOT/onpolicy-trace"
ON_LAND="$ROOT/onpolicy-landmarks"
mkdir -p "$CANON_LAND" "$OFF_LAND" "$ON_TRACE_DIR" "$ON_LAND"

phase "learn fixed start and off-policy vocabularies"
.venv/bin/python scripts/learn_global_navigation_landmarks.py \
  --trace "$MEDOID_TRACE" \
  --portals "$PORTALS" \
  --out-dir "$CANON_LAND" \
  --budgets "16000" \
  2>&1 | tee "$ART/learn-16k.log"

.venv/bin/python scripts/learn_staged_navigation_landmarks.py \
  --trace "$MEDOID_TRACE" \
  --out-dir "$OFF_LAND" \
  --budget 8000 \
  --stages "0,$STAGE" \
  2>&1 | tee "$ART/learn-offpolicy.log"
cp "$OFF_LAND/staged-landmarks.manifest.json" "$ART/offpolicy-landmarks.manifest.json"

build_ivf() {
  local hints="$1"
  local out="$2"
  local nlist="$3"
  local log="$4"
  rm -f "$out" "$out.manifest.json"
  .venv/bin/python scripts/build_hint_ivf.py \
    --base "$BASE" \
    --hints "$hints" \
    --out "$out" \
    --nlist "$nlist" \
    --iterations 5 \
    --batch 2048 \
    --seed 20261005 \
    2>&1 | tee "$ART/$log"
}

CANON_IVF="$ROOT/canonical-b16000-c512.bin"
START_IVF="$ROOT/stage0-b8000-c256.bin"
OFF_STAGE_IVF="$ROOT/offpolicy-stage${STAGE}-b8000-c256.bin"
ON_STAGE_IVF="$ROOT/onpolicy-stage${STAGE}-b8000-c256.bin"

phase "build fixed directories"
build_ivf "$CANON_LAND/landmarks-b16000.bin" "$CANON_IVF" 512 "build-16k.log"
build_ivf "$OFF_LAND/stage0-b8000.bin" "$START_IVF" 256 "build-start8k.log"
build_ivf "$OFF_LAND/stage${STAGE}-b8000.bin" "$OFF_STAGE_IVF" 256 "build-offpolicy-stage${STAGE}.log"

phase "install pinned Rust toolchain"
export CARGO_HOME="$RUNNER_TEMP/onpolicy-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/onpolicy-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/onpolicy-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/onpolicy-rustup-$GITHUB_RUN_ID.sh" \
  -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

phase "build optimized NavHints trace binary"
rm -rf third_party/DiskANN-onpolicy-trace
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-onpolicy-trace
git -C third_party/DiskANN-onpolicy-trace checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_navhints_trace.py third_party/DiskANN-onpolicy-trace
(cd third_party/DiskANN-onpolicy-trace && cargo fmt --all)
git -C third_party/DiskANN-onpolicy-trace diff --check
git -C third_party/DiskANN-onpolicy-trace diff > "$ART/onpolicy-trace.patch"
(cd third_party/DiskANN-onpolicy-trace && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/trace-build.log"
TRACE_BIN="$PWD/third_party/DiskANN-onpolicy-trace/target/release/diskann-benchmark"

phase "collect 8K-start on-policy L4 traces"
TRACE_WORK="$RUNNER_TEMP/onpolicy-trace-eval-$GITHUB_RUN_ID"
rm -rf "$TRACE_WORK" "$ON_TRACE_DIR"
mkdir -p "$TRACE_WORK" "$ON_TRACE_DIR"
.venv/bin/python scripts/collect_navhints_onpolicy_traces.py \
  --binary "$TRACE_BIN" \
  --queries "$QUERIES" \
  --gt "$GT" \
  --index-prefix "$INDEX_PREFIX" \
  --start-ivf "$START_IVF" \
  --work "$TRACE_WORK" \
  --out "$ON_TRACE_DIR" \
  --train-rows 5000 \
  2>&1 | tee "$ART/onpolicy-trace.log"
cp "$ON_TRACE_DIR/onpolicy-trace-summary.json" "$ART/"
sha256sum "$ON_TRACE_DIR/navhints8k-teacher-L4.jsonl" > "$ART/onpolicy-trace.sha256"
rm -rf "$TRACE_WORK"

phase "learn on-policy hop-${STAGE} vocabulary"
rm -rf "$ON_LAND"
mkdir -p "$ON_LAND"
.venv/bin/python scripts/learn_staged_navigation_landmarks.py \
  --trace "$ON_TRACE_DIR/navhints8k-teacher-L4.jsonl" \
  --out-dir "$ON_LAND" \
  --budget 8000 \
  --stages "$STAGE" \
  2>&1 | tee "$ART/learn-onpolicy.log"
cp "$ON_LAND/staged-landmarks.manifest.json" "$ART/onpolicy-landmarks.manifest.json"
build_ivf "$ON_LAND/stage${STAGE}-b8000.bin" "$ON_STAGE_IVF" 256 "build-onpolicy-stage${STAGE}.log"

phase "compare off-policy and on-policy vocabularies"
STAGE="$STAGE" .venv/bin/python - <<'PY' > "$ART/stage${STAGE}-overlap.json"
import json, struct
from pathlib import Path
import numpy as np

MAGIC=b"GIDST001"
def read(path):
    raw=Path(path).read_bytes()
    assert raw[:8]==MAGIC
    n,res=struct.unpack("<II",raw[8:16]); assert res==0
    return set(np.frombuffer(raw,dtype="<u4",count=n,offset=16).astype(int).tolist())

root=Path.home()/".cache/geoivf/onpolicy-staged-navhints-v1"
stage=int(__import__("os").environ["STAGE"])
off=read(root/f"offpolicy-landmarks/stage{stage}-b8000.bin")
on=read(root/f"onpolicy-landmarks/stage{stage}-b8000.bin")
start=read(root/"offpolicy-landmarks/stage0-b8000.bin")
def cmp(a,b):
    inter=len(a&b); union=len(a|b)
    return {"intersection":inter,"fraction_of_8k":inter/8000,"jaccard":inter/union}
print(json.dumps({
    f"onpolicy_vs_offpolicy_stage{stage}":cmp(on,off),
    f"onpolicy_stage{stage}_vs_start8k":cmp(on,start),
    f"offpolicy_stage{stage}_vs_start8k":cmp(off,start),
},indent=2))
PY

phase "build staged evaluation binary"
rm -rf third_party/DiskANN-onpolicy-staged
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-onpolicy-staged
git -C third_party/DiskANN-onpolicy-staged checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_staged_navhints.py third_party/DiskANN-onpolicy-staged
(cd third_party/DiskANN-onpolicy-staged && cargo fmt --all)
git -C third_party/DiskANN-onpolicy-staged diff --check
git -C third_party/DiskANN-onpolicy-staged diff > "$ART/staged-eval.patch"
(cd third_party/DiskANN-onpolicy-staged && cargo build --release --locked -p diskann-benchmark --features disk-index) \
  2>&1 | tee "$ART/eval-build.log"
EVAL_BIN="$PWD/third_party/DiskANN-onpolicy-staged/target/release/diskann-benchmark"

phase "run four-arm heldout comparison"
EVAL_WORK="$RUNNER_TEMP/onpolicy-staged-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
rm -rf "$EVAL_WORK"
mkdir -p "$EVAL_WORK" "$OUT"
.venv/bin/python scripts/qualify_onpolicy_staged_navhints.py \
  --binary "$EVAL_BIN" \
  --queries "$QUERIES" \
  --gt "$GT" \
  --index-prefix "$INDEX_PREFIX" \
  --ivf-16k "$CANON_IVF" \
  --ivf-start8k "$START_IVF" \
  --ivf-stage-offpolicy "$OFF_STAGE_IVF" \
  --ivf-stage-onpolicy "$ON_STAGE_IVF" \
  --stage-hops "$STAGE" \
  --work "$EVAL_WORK" \
  --out "$OUT" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"
rm -rf "$EVAL_WORK"
