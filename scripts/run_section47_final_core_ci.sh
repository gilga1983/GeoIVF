#!/usr/bin/env bash
set -euxo pipefail
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
else
 PUBLIC="$ROOT/public-eval/bigann-10M"
 DATA="$PUBLIC/data"
 MANIFEST="$DATA/dataset.manifest.json"
 test -s "$MANIFEST"
 BASE="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["base"])' "$MANIFEST")"
 QUERIES="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["heldout5000"])' "$MANIFEST")"
 GT="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["files"]["heldout5000_gt"])' "$MANIFEST")"
 INDEX="$PUBLIC/index/diskann-index"
 TRACE="$PUBLIC/trace/teacher.jsonl"
 IVF="$PUBLIC/navhints/hints-b14464-nlist512.bin"
 DTYPE="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["data_type"])' "$MANIFEST")"
 METRIC="$("$PYC" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metric"])' "$MANIFEST")"
fi
for file in "$BASE" "$QUERIES" "$GT" "$TRACE" "$IVF"; do
 if [[ ! -s "$file" ]]; then echo "Missing required $DATASET input: $file" >&2; exit 1; fi
done
# Public BigANN has a .complete manifest, while the already-qualified
# PubMed index is tracked by its native DiskANN disk-graph file.
"$PYC" - "$INDEX" <<'PY'
from pathlib import Path
import sys
prefix=Path(sys.argv[1])
matches=[p for p in prefix.parent.glob(prefix.name+"*")
         if p.is_file() and ("disk.index" in p.name or p.name.endswith("_disk.index"))]
if len(matches)!=1 or not matches[0].stat().st_size:
    raise SystemExit(f"Missing or ambiguous native DiskANN index for {prefix}: {matches}")
print("Verified native DiskANN graph:",matches[0],flush=True)
PY
# Charge cached graph records by their actual vector type (uint8 on BigANN),
# rather than using float32 bytes for both datasets.
BUDGETS=$("$PYC" - "$IVF" "$BASE" "$DTYPE" <<'PY'
from pathlib import Path
import struct,sys
ivf,base=map(Path,sys.argv[1:3]);dtype=sys.argv[3]
rows,dim=struct.unpack("<II",base.open("rb").read(8))
expected=ivf.stat().st_size+512*(64+4)+2048*4
if expected!=102940:
    raise SystemExit(f"Unexpected final-Core payload: {expected}")
if dtype not in ("float32","uint8"):raise SystemExit("Unexpected vector type")
record=dim*(4 if dtype=="float32" else 1)+64*4
scales=(1,4,16)
print(",".join(str(expected*s//record) for s in scales))
print(",".join(str(expected*s//(dim*4+4)) for s in scales))
print(f"Final Core={expected}B; graph record={record}B; rows={rows}",file=sys.stderr)
PY
)
HOT_COUNTS="$(sed -n '1p' <<< "$BUDGETS")"
QSEV_COUNTS="$(sed -n '2p' <<< "$BUDGETS")"
IFS=',' read -r -a QSEV_ARRAY <<< "$QSEV_COUNTS"
printf 'dataset=%s core_bytes=102940 cache_nodes=%s qsev_entries=%s\n' \
 "$DATASET" "$HOT_COUNTS" "$QSEV_COUNTS" | tee "$ART/budget-contract.txt"
sha256sum "$IVF" "$TRACE" > "$ART/input-hashes.txt"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ uname -a; free -h; df -h; } > "$ART/host.txt"
"$PYC" scripts/learn_hot_cache_ids.py --trace "$TRACE" --out-dir "$WORK/hot" --counts "$HOT_COUNTS" > "$ART/train-hot.log" 2>&1
if [[ "$DATASET" == pubmed ]]; then
 for N in "${QSEV_ARRAY[@]}";do
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
 (cd "$DIR";git diff --check;cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index) > "$ART/build-$flavor.log" 2>&1
done
ARGS=(--dataset "$DATASET" --index-prefix "$INDEX" --queries "$QUERIES" --gt "$GT" --ivf "$IVF" --core-binary "$WORK/diskann-core/target/release/diskann-benchmark" --cache-binary "$WORK/diskann-cache/target/release/diskann-benchmark" --hot-dir "$WORK/hot" --out "$ART/results" --data-type "$DTYPE" --metric "$METRIC" --reps 3)
if [[ "$DATASET" == pubmed ]];then ARGS+=(--qsev-binary "$WORK/diskann-qsev/target/release/diskann-benchmark" --qsev-dir "$WORK");fi
"$PYC" scripts/qualify_section47_final_core.py "${ARGS[@]}" > "$ART/eval.log" 2>&1
test -s "$ART/results/section47-final-core.json"
echo "SECTION47_${DATASET}_SUCCESS"
