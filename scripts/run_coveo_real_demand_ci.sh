#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
RAW="$HOME/.cache/geoivf/coveo-sigir-ecom-2021/raw"
SEARCH="$RAW/search_train.csv"
CATALOG="$RAW/sku_to_content.csv"
ROOT="$HOME/.cache/geoivf/coveo-real-demand-v1"
PREP="$ROOT/prepared"
SPLIT="$ROOT/split"
INDEX_ROOT="$ROOT/index"
INDEX_PREFIX="$INDEX_ROOT/diskann-index"
TRACE_ROOT="$ROOT/trace"
TRACE="$TRACE_ROOT/teacher.jsonl"
STATE="$ROOT/navhints"
LANDMARKS="$STATE/landmarks-b16000.bin"
IVF="$STATE/hints-b16000-nlist512.bin"
ART="$PWD/artifacts/coveo-real-demand"
WORK="$RUNNER_TEMP/coveo-real-demand-$GITHUB_RUN_ID"

mkdir -p "$ART" "$PREP" "$SPLIT" "$INDEX_ROOT" "$TRACE_ROOT" "$STATE" "$WORK"

if [ ! -s "$SEARCH" ] || [ ! -s "$CATALOG" ]; then
  cat >&2 <<EOF
Coveo full research dataset is not installed.
After accepting the Coveo SIGIR eCom 2021 research terms, place:
  $SEARCH
  $CATALOG
The public-sample adapter has already been validated; this workflow intentionally
does not bypass Coveo's access form or redistribute the licensed archive.
EOF
  exit 3
fi

git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.2.6

python scripts/prepare_coveo_real_demand.py \
  --search-csv "$SEARCH" \
  --catalog-csv "$CATALOG" \
  --out-dir "$PREP" \
  2>&1 | tee "$ART/prepare.log"

python scripts/freeze_coveo_paper_split.py \
  --prepared-dir "$PREP" \
  --out-dir "$SPLIT" \
  --start 0 \
  --static-train 5000 \
  --online-warmup 20000 \
  --eval 5000 \
  2>&1 | tee "$ART/freeze-split.log"

BASE="$PREP/coveo-products-cosine.fbin"
ALLQ="$PREP/coveo-search-chronological-cosine.fbin"
TRAIN="$SPLIT/train.fbin"
REPLAY="$SPLIT/replay.fbin"
EVAL="$SPLIT/eval.fbin"
ALL_GT="$SPLIT/first30000.gt"
TRAIN_GT="$SPLIT/train.gt"
REPLAY_GT="$SPLIT/replay.gt"
EVAL_GT="$SPLIT/eval.gt"

python scripts/build_exact_ip_gt.py \
  --base "$BASE" --queries "$ALLQ" --out "$ALL_GT" \
  --start 0 --count 30000 --topk 16 --query-batch 512 \
  2>&1 | tee "$ART/exact-gt.log"

python - "$ALL_GT" "$TRAIN_GT" "$REPLAY_GT" "$EVAL_GT" <<'PY'
import struct,sys
from pathlib import Path
src,train,replay,ev=map(Path,sys.argv[1:])
with src.open("rb") as f:
    rows,k=struct.unpack("<II",f.read(8))
    assert rows==30000
    ids=f.read(rows*k*4)
    dists=f.read(rows*k*4)
    assert len(ids)==rows*k*4 and len(dists)==rows*k*4
def out(path,start,count):
    rb=k*4
    with path.open("wb") as g:
        g.write(struct.pack("<II",count,k))
        g.write(ids[start*rb:(start+count)*rb])
        g.write(dists[start*rb:(start+count)*rb])
out(train,0,5000)
out(replay,5000,25000)
out(ev,25000,5000)
PY

python - "$SPLIT/dataset.manifest.json" "$EVAL_GT" "$SPLIT/build.manifest.json" <<'PY'
import json,sys
from pathlib import Path
src,gt,out=map(Path,sys.argv[1:])
m=json.loads(src.read_text())
m["files"]["heldout5000"]=m["files"]["eval"]
m["files"]["heldout5000_gt"]=str(gt.resolve())
out.write_text(json.dumps(m,indent=2)+"\n")
PY
BUILD_MANIFEST="$SPLIT/build.manifest.json"

export CARGO_HOME="$RUNNER_TEMP/coveo-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/coveo-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/coveo-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/coveo-rustup-$GITHUB_RUN_ID.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-trace"
git -C "$WORK/DiskANN-trace" checkout --detach "$DISKANN_REV"
python scripts/patch_diskann_waypoint_trace.py "$WORK/DiskANN-trace"
(
  cd "$WORK/DiskANN-trace"
  cargo fmt --all
  git diff --check
  cargo build --release --locked -p diskann-benchmark --features disk-index
) 2>&1 | tee "$ART/build-trace.log"
TRACE_BIN="$WORK/DiskANN-trace/target/release/diskann-benchmark"

if [ ! -s "$INDEX_PREFIX.complete.json" ]; then
  rm -f "$INDEX_PREFIX.complete.json"
  python scripts/build_public_diskann_index.py \
    --binary "$TRACE_BIN" \
    --dataset-manifest "$BUILD_MANIFEST" \
    --save-prefix "$INDEX_PREFIX" \
    --work "$WORK/index-build" \
    --build-ram-gb 32 \
    2>&1 | tee "$ART/index-build.log"
fi
test -s "$INDEX_PREFIX.complete.json"

rm -rf "$TRACE_ROOT"
mkdir -p "$TRACE_ROOT"
python scripts/collect_navigation_trace_generic.py \
  --binary "$TRACE_BIN" \
  --queries "$TRAIN" \
  --gt "$TRAIN_GT" \
  --index-prefix "$INDEX_PREFIX" \
  --data-type float32 \
  --distance inner_product \
  --train-rows 5000 \
  --teacher-l 4 \
  --work "$WORK/trace" \
  --out "$TRACE_ROOT" \
  2>&1 | tee "$ART/trace.log"
test -s "$TRACE"

rm -rf "$STATE"
mkdir -p "$STATE"
python scripts/learn_navhints_generic.py \
  --trace "$TRACE" \
  --out "$LANDMARKS" \
  --budget 16000 \
  2>&1 | tee "$ART/learn.log"

python scripts/build_hint_ivf_generic.py \
  --base "$BASE" \
  --base-data-type float32 \
  --hints "$LANDMARKS" \
  --out "$IVF" \
  --partition spherical \
  --nlist 512 \
  --iterations 5 \
  --batch 1024 \
  --seed 20261005 \
  2>&1 | tee "$ART/build-ivf.log"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN-search"
git -C "$WORK/DiskANN-search" checkout --detach "$DISKANN_REV"
python scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/DiskANN-search"
python scripts/patch_diskann_experience_core.py "$WORK/DiskANN-search"
(
  cd "$WORK/DiskANN-search"
  cargo fmt --all
  git diff --check
  cargo build --release --locked -p diskann-benchmark --features disk-index
) 2>&1 | tee "$ART/build-search.log"
BIN="$WORK/DiskANN-search/target/release/diskann-benchmark"

mkdir -p "$ART/results"
python scripts/qualify_coveo_real_demand.py \
  --binary "$BIN" \
  --dataset-manifest "$SPLIT/dataset.manifest.json" \
  --replay-gt "$REPLAY_GT" \
  --index-prefix "$INDEX_PREFIX" \
  --ivf "$IVF" \
  --work "$WORK/eval" \
  --out "$ART/results" \
  --reps 3 \
  2>&1 | tee "$ART/eval.log"

cp "$PREP/coveo-ann.manifest.json" "$ART/"
cp "$SPLIT/dataset.manifest.json" "$ART/frozen-split.manifest.json"
cp "$LANDMARKS.manifest.json" "$ART/landmarks.manifest.json"
cp "$IVF.manifest.json" "$ART/ivf.manifest.json"
cp "$INDEX_PREFIX.complete.json" "$ART/index.complete.json"
rm -rf "$WORK"
