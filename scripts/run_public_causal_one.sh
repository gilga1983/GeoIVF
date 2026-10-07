#!/usr/bin/env bash
set -euxo pipefail

DATASET="${1:?usage: run_public_causal_one.sh DATASET ARTIFACT_DIR}"
ARTIFACT_DIR="${2:?usage: run_public_causal_one.sh DATASET ARTIFACT_DIR}"
DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

case "$DATASET" in
  text2image-10M) PARTITION=spherical ;;
  bigann-10M|bigann-100M) PARTITION=l2 ;;
  *) echo "unsupported dataset $DATASET" >&2; exit 2 ;;
esac

ROOT="$HOME/.cache/geoivf/public-eval/$DATASET"
DATA="$ROOT/data"
INDEX_ROOT="$ROOT/index"
INDEX_PREFIX="$INDEX_ROOT/diskann-index"
TRACE_ROOT="$ROOT/trace"
TRACE="$TRACE_ROOT/teacher.jsonl"
STATE="$ROOT/navhints"
LANDMARKS="$STATE/landmarks-b16000.bin"
IVF="$STATE/hints-b16000-nlist512.bin"
WORK="$RUNNER_TEMP/public-causal-${DATASET}-$GITHUB_RUN_ID"

mkdir -p "$ARTIFACT_DIR" "$DATA" "$INDEX_ROOT" "$TRACE_ROOT" "$STATE" "$WORK"
git rev-parse HEAD > "$ARTIFACT_DIR/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ARTIFACT_DIR/host.txt"

python3 scripts/prepare_public_ann_dataset.py --dataset "$DATASET" --out "$DATA" 2>&1 | tee "$ARTIFACT_DIR/prepare-data.log"
cp "$DATA/dataset.manifest.json" "$ARTIFACT_DIR/"

DATA_TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_type"])' "$DATA/dataset.manifest.json")"
DISTANCE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metric"])' "$DATA/dataset.manifest.json")"
BASE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["base"])' "$DATA/dataset.manifest.json")"
QUERIES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["queries"])' "$DATA/dataset.manifest.json")"
HELD_GT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["heldout5000_gt"])' "$DATA/dataset.manifest.json")"

export CARGO_HOME="$RUNNER_TEMP/public-causal-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/public-causal-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/public-causal-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/public-causal-rustup-$GITHUB_RUN_ID.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

# Reuse the exact public data/index/static-routing preparation from the verified suite.
if [ ! -s "$INDEX_PREFIX.complete.json" ] || [ ! -s "$TRACE" ] || [ ! -s "$LANDMARKS" ] || [ ! -s "$IVF" ]; then
  git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-trace"
  git -C "$WORK/DiskANN-trace" checkout --detach "$DISKANN_REV"
  python3 scripts/patch_diskann_waypoint_trace.py "$WORK/DiskANN-trace"
  (cd "$WORK/DiskANN-trace" && cargo fmt --all && cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ARTIFACT_DIR/build-trace-binary.log"
  TRACE_BIN="$WORK/DiskANN-trace/target/release/diskann-benchmark"

  if [ ! -s "$INDEX_PREFIX.complete.json" ]; then
    python3 scripts/build_public_diskann_index.py --binary "$TRACE_BIN" --dataset-manifest "$DATA/dataset.manifest.json" --save-prefix "$INDEX_PREFIX" --work "$WORK/index-build" --build-ram-gb 32 2>&1 | tee "$ARTIFACT_DIR/index-build.log"
  fi
  if [ ! -s "$TRACE" ]; then
    rm -rf "$TRACE_ROOT"; mkdir -p "$TRACE_ROOT"
    python3 scripts/collect_navigation_trace_generic.py --binary "$TRACE_BIN" --queries "$QUERIES" --gt "$HELD_GT" --index-prefix "$INDEX_PREFIX" --data-type "$DATA_TYPE" --distance "$DISTANCE" --train-rows 5000 --teacher-l 4 --work "$WORK/trace" --out "$TRACE_ROOT" 2>&1 | tee "$ARTIFACT_DIR/trace.log"
  fi
fi

test -s "$INDEX_PREFIX.complete.json"
test -s "$TRACE"
if [ ! -s "$LANDMARKS" ]; then
  python3 scripts/learn_navhints_generic.py --trace "$TRACE" --out "$LANDMARKS" --budget 16000 2>&1 | tee "$ARTIFACT_DIR/learn.log"
fi
if [ ! -s "$IVF" ]; then
  python3 scripts/build_hint_ivf_generic.py --base "$BASE" --base-data-type "$DATA_TYPE" --hints "$LANDMARKS" --out "$IVF" --partition "$PARTITION" --nlist 512 --iterations 5 --batch 1024 --seed 20261005 2>&1 | tee "$ARTIFACT_DIR/build-ivf.log"
fi
cp "$INDEX_PREFIX.complete.json" "$ARTIFACT_DIR/"
cp "$LANDMARKS.manifest.json" "$ARTIFACT_DIR/landmarks.manifest.json"
cp "$IVF.manifest.json" "$ARTIFACT_DIR/ivf.manifest.json"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-search"
git -C "$WORK/DiskANN-search" checkout --detach "$DISKANN_REV"
python3 scripts/patch_diskann_vertex_navhints.py "$WORK/DiskANN-search" --variants 5
python3 scripts/patch_diskann_experience_controller.py "$WORK/DiskANN-search"
(cd "$WORK/DiskANN-search" && cargo fmt --all && git diff --check && cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ARTIFACT_DIR/build-search-binary.log"
BIN="$WORK/DiskANN-search/target/release/diskann-benchmark"

mkdir -p "$ARTIFACT_DIR/results"
source scripts/perf_guard.sh
geoivf_perf_lock
python3 scripts/qualify_public_causal_fill2.py --binary "$BIN" --dataset-manifest "$DATA/dataset.manifest.json" --index-prefix "$INDEX_PREFIX" --ivf "$IVF" --work "$WORK/eval" --out "$ARTIFACT_DIR/results" --reps 3 2>&1 | tee "$ARTIFACT_DIR/eval.log"

rm -rf "$WORK"
