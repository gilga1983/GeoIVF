#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/temporal-portfolio"
ROOT="$HOME/.cache/geoivf/temporal-portfolio-v1"
ONPOL="$HOME/.cache/geoivf/onpolicy-staged-navhints-v1"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6
mkdir -p "$ART" "$ROOT"

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$HOME/.cache/geoivf/high-l-medoid-teacher-traces-v1/medoid-teacher.L64.jsonl"
GENERAL_IDS="$ONPOL/offpolicy-landmarks/stage0-b8000.bin"
GENERAL_IVF="$ONPOL/stage0-b8000-c256.bin"
CANON_IDS="$ONPOL/canonical-landmarks/landmarks-b16000.bin"
CANON_IVF="$ONPOL/canonical-b16000-c512.bin"
for f in "$BASE" "$QUERIES" "$GT" "$TRACE" "$GENERAL_IDS" "$GENERAL_IVF" "$CANON_IDS" "$CANON_IVF"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
.venv/bin/python -m py_compile   scripts/learn_temporal_banks.py   scripts/build_hint_ivf.py   scripts/patch_diskann_temporal_portfolio.py   scripts/qualify_temporal_portfolio.py

build_banks() {
  local tag="$1"
  local exclude="$2"
  local dir="$ROOT/$tag"
  rm -rf "$dir"
  mkdir -p "$dir"
  .venv/bin/python scripts/learn_temporal_banks.py     --trace "$TRACE" --exclude-ids "$exclude" --out-dir "$dir"     --stages "8,16,24" --budgets "4000,2000,2000"     2>&1 | tee "$ART/learn-$tag.log"
  cp "$dir/manifest.json" "$ART/$tag-banks.manifest.json"

  for spec in "8 4000 128" "16 2000 64" "24 2000 64"; do
    set -- $spec
    stage="$1"; budget="$2"; nlist="$3"
    ids="$dir/stage${stage}-b${budget}.bin"
    out="$dir/stage${stage}-b${budget}-c${nlist}.ivf"
    .venv/bin/python scripts/build_hint_ivf.py       --base "$BASE" --hints "$ids" --out "$out"       --nlist "$nlist" --iterations 5 --batch 2048 --seed 20261006       2>&1 | tee "$ART/build-$tag-stage${stage}.log"
    cp "$out.manifest.json" "$ART/$tag-stage${stage}.ivf.manifest.json"
  done
}

build_banks partition "$GENERAL_IDS"
build_banks additive "$CANON_IDS"

export CARGO_HOME="$RUNNER_TEMP/temporalpf-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/temporalpf-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/temporalpf-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/temporalpf-rustup-$GITHUB_RUN_ID.sh"   -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-temporal-portfolio
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-temporal-portfolio
git -C third_party/DiskANN-temporal-portfolio checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_temporal_portfolio.py third_party/DiskANN-temporal-portfolio
(cd third_party/DiskANN-temporal-portfolio && cargo fmt --all)
git -C third_party/DiskANN-temporal-portfolio diff --check
git -C third_party/DiskANN-temporal-portfolio diff > "$ART/temporal-portfolio.patch"
(cd third_party/DiskANN-temporal-portfolio && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee "$ART/build.log"
BIN="$PWD/third_party/DiskANN-temporal-portfolio/target/release/diskann-benchmark"

WORK="$RUNNER_TEMP/temporalpf-eval-$GITHUB_RUN_ID"
OUT="$ART/results"
mkdir -p "$WORK" "$OUT"
.venv/bin/python scripts/qualify_temporal_portfolio.py   --binary "$BIN" --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --canonical "$CANON_IVF" --general8 "$GENERAL_IVF"   --partition-s8 "$ROOT/partition/stage8-b4000-c128.ivf"   --partition-s16 "$ROOT/partition/stage16-b2000-c64.ivf"   --partition-s24 "$ROOT/partition/stage24-b2000-c64.ivf"   --additive-s8 "$ROOT/additive/stage8-b4000-c128.ivf"   --additive-s16 "$ROOT/additive/stage16-b2000-c64.ivf"   --additive-s24 "$ROOT/additive/stage24-b2000-c64.ivf"   --work "$WORK" --out "$OUT" --reps 3   2>&1 | tee "$ART/eval.log"

rm -rf "$WORK"
