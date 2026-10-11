#!/usr/bin/env bash
# Measure Table 7 controls on Coveo and Text2Image only.
set -euxo pipefail
DATASET="$1"
ROOT="$HOME/.cache/geoivf"
ART="$PWD/artifacts/table7-$DATASET"
WORK="$RUNNER_TEMP/table7-$DATASET-$GITHUB_RUN_ID"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
mkdir -p "$ART" "$WORK"
trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install numpy==2.2.6 'faiss-cpu>=1.9,<2' > "$ART/pip.log" 2>&1
if [[ "$DATASET" == coveo ]]; then
 DIR="$ROOT/coveo-real-demand-v1"
 BASE="$DIR/prepared/coveo-products-cosine.fbin"
 QUERIES="$DIR/split/replay.fbin"
 GT="$DIR/split/replay.gt"
 INDEX="$DIR/index/diskann-index"
 TRACE="$DIR/trace/teacher.jsonl"
 IVF="$DIR/navhints-final-core-20261011/hints-b14464-nlist512.bin"
elif [[ "$DATASET" == text2image-10M ]]; then
 DIR="$ROOT/public-eval/text2image-10M"
 MANIFEST="$DIR/data/dataset.manifest.json"
 test -s "$MANIFEST"
 BASE="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["base"])' "$MANIFEST")"
 QUERIES="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["heldout5000"])' "$MANIFEST")"
 GT="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["heldout5000_gt"])' "$MANIFEST")"
 INDEX="$DIR/index/diskann-index"
 TRACE="$DIR/trace/teacher.jsonl"
 IVF="$DIR/navhints/hints-b14464-nlist512.bin"
else
 echo "Previously measured PubMed/BigANN competitors must NOT be rerun" >&2
 exit 2
fi
for FILE in "$BASE" "$QUERIES" "$GT" "$TRACE" "$IVF"; do
 if [[ ! -s "$FILE" ]]; then echo "Missing frozen input: $FILE" >&2;exit 1;fi
done
test -s "$INDEX.complete.json"
sha256sum "$BASE" "$QUERIES" "$GT" "$IVF" "$TRACE" > "$ART/input-sha256.txt"
git rev-parse HEAD > "$ART/geoivf-source-sha.txt"
{ uname -a; free -h; df -h; } > "$ART/host.txt"
BUDGETS="$("$PYC" - "$IVF" "$BASE" <<'PY'
import sys,struct
from pathlib import Path
ivf,base=map(Path,sys.argv[1:3])
rows,dim=struct.unpack("<II",base.open("rb").read(8))
payload=ivf.stat().st_size+512*(64+4)+2048*4
cache=[payload*s//(dim*4+64*4) for s in (1,4,16)]
qsev=[(payload*s-16)//(dim*4+4) for s in (1,4,16)]
print(",".join(map(str,cache)))
print(",".join(map(str,qsev)))
print("Core bytes",payload,"dim",dim,"rows",rows,file=sys.stderr)
PY
)"
HOT_COUNTS="$(sed -n '1p' <<< "$BUDGETS")"
QSEV_COUNTS="$(sed -n '2p' <<< "$BUDGETS")"
printf 'dataset=%s hot_bfs=%s qsev=%s\n' "$DATASET" "$HOT_COUNTS" "$QSEV_COUNTS" > "$ART/budgets.txt"
"$PYC" scripts/learn_hot_cache_ids.py --trace "$TRACE" \
 --out-dir "$WORK/hot" --counts "$HOT_COUNTS" > "$ART/prepare-hot.log" 2>&1
for N in $(tr ',' ' ' <<< "$QSEV_COUNTS"); do
 "$PYC" scripts/prepare_diskannpp_qsev.py \
 --base "$BASE" --index-prefix "$INDEX" \
 --out "$WORK/qsev-$N.bin" --clusters "$((N-1))" \
 --train-size 100000 --threads 2 --seed 12345 \
 > "$ART/prepare-qsev-$N.log" 2>&1
done
export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1
for MODE in cache qsev; do
 SRC="$WORK/diskann-$MODE"
 git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$SRC" > "$ART/clone-$MODE.log" 2>&1
 git -C "$SRC" checkout --detach "$REV"
 if [[ "$MODE" == cache ]]; then
  "$PYC" scripts/patch_diskann_hot_cache.py "$SRC"
 else
  "$PYC" scripts/patch_diskann_qsev.py "$SRC"
 fi
 git -C "$SRC" diff --check
 (cd "$SRC"; CARGO_BUILD_JOBS=2 cargo +1.97.1 build --release --locked \
   -p diskann-benchmark --features disk-index) > "$ART/build-$MODE.log" 2>&1
done
"$PYC" scripts/qualify_table7_missing_workloads.py \
 --dataset "$DATASET" --reference repro/table7_frozen_core_reference.json \
 --index-prefix "$INDEX" --queries "$QUERIES" --gt "$GT" --ivf "$IVF" \
 --cache-binary "$WORK/diskann-cache/target/release/diskann-benchmark" \
 --qsev-binary "$WORK/diskann-qsev/target/release/diskann-benchmark" \
 --hot-dir "$WORK/hot" --qsev-dir "$WORK" --out "$ART/results" \
 --reps 3 > "$ART/eval.log" 2>&1
test -s "$ART/results/table7-first-time-results.json"
printf 'TABLE7_FIRST_TIME_%s_SUCCESS\n' "$DATASET"
