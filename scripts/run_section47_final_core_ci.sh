#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?pubmed or bigann-10M}"
ROOT="$HOME/.cache/geoivf"
WORK="$RUNNER_TEMP/section47-$DATASET-$GITHUB_RUN_ID"
ART="$PWD/artifacts/section47-$DATASET"
mkdir -p "$WORK" "$ART"
trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install numpy==2.2.6 'faiss-cpu>=1.9,<2' > "$ART/pip.log" 2>&1
PYC="$WORK/venv/bin/python"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1
if [[ "$DATASET" == pubmed ]]; then
 BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
 ALLQ="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
 GT="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
 INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
 TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
 STATE="$ROOT/navhints-fixed-memory-tradeoff-20261011"
 HINTS="$STATE/landmarks/landmarks-b14464.bin"
 IVF="$STATE/ivf/hints-b14464-nlist512-spherical.bin"
 METRIC=inner_product; DTYPE=float32
 # Distinct 5000 test requests (online warmup 4000, measured 1000).
 "$PYC" - "$ALLQ" "$WORK/heldout.fbin" <<'PY'
import struct,sys
with open(sys.argv[1],"rb") as f:
 n,d=struct.unpack("<II",f.read(8))
 assert (n,d)==(10000,768)
 f.seek(8+5000*d*4);body=f.read(5000*d*4)
 assert len(body)==5000*d*4
with open(sys.argv[2],"wb") as f:f.write(struct.pack("<II",5000,d)+body)
PY
 QUERIES="$WORK/heldout.fbin"
 HOT_COUNTS=30,120,480
else
 PUBLIC="$ROOT/public-eval/bigann-10M"
 DATA="$PUBLIC/data"
 MANIFEST="$DATA/dataset.manifest.json"
 test -s "$MANIFEST"
 BASE="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["base"])' "$MANIFEST")"
 QUERIES="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["queries"])' "$MANIFEST")"
 GT="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["heldout5000_gt"])' "$MANIFEST")"
 INDEX="$PUBLIC/index/diskann-index"
 TRACE="$PUBLIC/trace/teacher.jsonl"
 IVF="$PUBLIC/navhints/hints-b14464-nlist512.bin"
 DTYPE="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["data_type"])' "$MANIFEST")"
 METRIC="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metric"])' "$MANIFEST")"
 HOT_COUNTS=134,536,2144
fi
for file in "$BASE" "$QUERIES" "$GT" "$TRACE" "$IVF";do test -s "$file";done
test -s "$INDEX.complete.json"
sha256sum "$IVF" "$TRACE" > "$ART/input-hashes.txt"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; free -h; df -h; } > "$ART/host.txt"
"$PYC" scripts/learn_hot_cache_ids.py --trace "$TRACE" --out-dir "$WORK/hot" --counts "$HOT_COUNTS" > "$ART/train-hot.log" 2>&1
if [[ "$DATASET" == pubmed ]]; then
 for N in 32 128 512;do
  "$PYC" scripts/prepare_diskannpp_qsev.py --base "$BASE" --index-prefix "$INDEX" --out "$WORK/qsev-$N.bin" --clusters "$((N-1))" --train-size 100000 --seed 12345 --threads 4 > "$ART/train-qsev-$N.log" 2>&1
 done
fi
for flavor in core cache qsev;do
 if [[ "$flavor" == qsev && "$DATASET" != pubmed ]];then continue;fi
 DIR="$WORK/diskann-$flavor"
 git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$DIR" > "$ART/clone-$flavor.log" 2>&1
 git -C "$DIR" checkout --detach "$REV"
 case "$flavor" in
  core)
   "$PYC" scripts/patch_diskann_hint_ivf_packed_direct.py "$DIR"
   "$PYC" scripts/patch_diskann_experience_core.py "$DIR"
   "$PYC" scripts/patch_navhints_regional_history.py "$DIR";;
  cache) "$PYC" scripts/patch_diskann_hot_cache.py "$DIR";;
  qsev) "$PYC" scripts/patch_diskann_qsev.py "$DIR";;
 esac
 (cd "$DIR";cargo +1.97.1 fmt --all;git diff --check;cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index) > "$ART/build-$flavor.log" 2>&1
done
ARGS=(--dataset "$DATASET" --index-prefix "$INDEX" --queries "$QUERIES" --gt "$GT" --ivf "$IVF" --core-binary "$WORK/diskann-core/target/release/diskann-benchmark" --cache-binary "$WORK/diskann-cache/target/release/diskann-benchmark" --hot-dir "$WORK/hot" --out "$ART/results" --data-type "$DTYPE" --metric "$METRIC" --reps 3)
if [[ "$DATASET" == pubmed ]];then ARGS+=(--qsev-binary "$WORK/diskann-qsev/target/release/diskann-benchmark" --qsev-dir "$WORK");fi
"$PYC" scripts/qualify_section47_final_core.py "${ARGS[@]}" > "$ART/eval.log" 2>&1
test -s "$ART/results/section47-final-core.json"
echo "SECTION47_$DATASET_SUCCESS"
