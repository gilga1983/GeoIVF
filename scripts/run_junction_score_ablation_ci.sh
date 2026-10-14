#!/usr/bin/env bash
# Compare junction ranking scores without rerunning any external competitors.
# Freeze existing 5K teacher trace, selected Core and real-SSD index.
set -euo pipefail
DATASET="${1:?usage: run_junction_score_ablation_ci.sh pubmed|bigann}"
ROOT="$HOME/.cache/geoivf"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/junction-score-ablation/$DATASET"
WORK="${RUNNER_TEMP:-/tmp}/junction-score-$DATASET-${GITHUB_RUN_ID:-manual}"
mkdir -p "$ART" "$WORK" "$ART/selection" "$ART/ivf" "$ART/results"
echo "$(git rev-parse HEAD)" > "$ART/geoivf-commit.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"

case "$DATASET" in
pubmed)
  BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
  ALLQ="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
  GT="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
  INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
  TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
  STATE="$ROOT/navhints-fixed-memory-tradeoff-20261011"
  REFERENCE="$STATE/landmarks/landmarks-b14464.bin"
  REF_IVF="$STATE/ivf/hints-b14464-nlist512-spherical.bin"
  ;;
bigann)
  CACHE="$ROOT/public-eval/bigann-10M"
  DATA_MANIFEST="$CACHE/data/dataset.manifest.json"
  INDEX="$CACHE/index/diskann-index"
  TRACE="$CACHE/trace/teacher.jsonl"
  STATE="$CACHE/navhints"
  REFERENCE="$STATE/landmarks-b14464.bin"
  REF_IVF="$STATE/hints-b14464-nlist512.bin"
  ;;
*) echo "Only pubmed or bigann supported" >&2; exit 2 ;;
esac

for item in "$TRACE" "$REFERENCE" "$REF_IVF" "$REF_IVF.manifest.json"; do
  test -s "$item" || { echo "MISSING_FROZEN_INPUT: $item" >&2; exit 10; }
done
test -d "$(dirname "$INDEX")"
sha256sum "$TRACE" "$REFERENCE" "$REF_IVF" > "$ART/frozen-state-sha256.txt"

if [ "$DATASET" = pubmed ]; then
  for item in "$BASE" "$ALLQ" "$GT"; do test -s "$item"; done
  DATA_MANIFEST="$ART/input-dataset-manifest.json"
  PYTHONPATH="$PWD/scripts" python3 - "$ALLQ" "$GT" "$DATA_MANIFEST" "$WORK/heldout5000.fbin" <<'PY'
import json,sys
from pathlib import Path
from qualify_navhints_catapult_headtohead import shape,slice_fbin
full,gt,manifest,dst=map(Path,sys.argv[1:])
if shape(full)!=(10000,768):raise ValueError("Wrong frozen PubMed query file")
slice_fbin(full,dst,5000,5000)
manifest.write_text(json.dumps({
 "dataset":"PubMed1M/MedCPT",
 "split":{"static_train_rows":5000,"online_warmup_rows":4000,"measured_rows":1000},
 "data_type":"float32","metric":"inner_product",
 "files":{"heldout5000":str(dst),"heldout5000_gt":str(gt)}
},indent=2)+"\n")
PY
else
  test -s "$DATA_MANIFEST"
  BASE="$(python3 - "$DATA_MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["files"]["base"])
PY
)"
  test -s "$BASE"
  cp "$DATA_MANIFEST" "$ART/input-dataset-manifest.json"
fi

python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install numpy==2.2.6 > "$ART/pip.log" 2>&1
if [ "$DATASET" = bigann ]; then
  python3 - "$ART/system-numpy.txt" <<'PY'
import sys, numpy as np
from pathlib import Path
Path(sys.argv[1]).write_text(
    f"python={sys.version}\nnumpy={np.__version__}\n"
    f"numpy_path={np.__file__}\n"
    + str(np.show_config(mode="dicts")) + "\n"
)
PY
fi
"$PYC" scripts/learn_junction_score_variants.py \
 --trace "$TRACE" --reference-sum "$REFERENCE" --out "$ART/selection" \
 > "$ART/selection/learn.log" 2>&1

for mode in sum count mean; do
  HINTS="$ART/selection/landmarks-b14464-$mode.bin"
  IVF="$ART/ivf/ivf-$mode.bin"
  if [ "$DATASET" = pubmed ]; then
    "$PYC" scripts/build_hint_ivf.py \
      --base "$BASE" --hints "$HINTS" --out "$IVF" \
      --nlist 512 --iterations 5 --batch 2048 --seed 20261005 \
      > "$ART/ivf/$mode.log" 2>&1
  else
    # The frozen BigANN IVF was originally constructed with system Python,
    # not this workflow's private numpy==2.2.6 virtual environment.
    # Keep the exact builder source, seed, inputs and numeric stack before
    # comparing against the original frozen directory byte-for-byte.
    python3 scripts/build_hint_ivf_generic.py \
      --base "$BASE" --base-data-type uint8 --hints "$HINTS" \
      --out "$IVF" --partition l2 \
      --nlist 512 --iterations 5 --batch 1024 --seed 20261005 \
      > "$ART/ivf/$mode.log" 2>&1
  fi
  test -s "$IVF"; test -s "$IVF.manifest.json"
  sha256sum "$HINTS" "$IVF" >> "$ART/variant-sha256.txt"
done
# Original algorithm must be reproduced exactly, including regional geometry.
cmp "$REFERENCE" "$ART/selection/landmarks-b14464-sum.bin"
if [ "$DATASET" = bigann ]; then
  # SHA from the published frozen BigANN run #38136365199, artifact 11704362848.
  echo "76149d8cdeda5ed78d952d62a247589fd88e8439c9c018ab5199020084a1716e  $REF_IVF" \
    | sha256sum -c -
fi
cmp "$REF_IVF" "$ART/ivf/ivf-sum.bin"
echo SELECTED_NAVHINTS_ORIGINAL_REPRODUCED > "$ART/repro.status"

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/diskann" > "$ART/clone.log" 2>&1
git -C "$WORK/diskann" checkout --detach "$REV" >> "$ART/clone.log" 2>&1
test "$(git -C "$WORK/diskann" rev-parse HEAD)" = "$REV"
"$PYC" scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/diskann"
"$PYC" scripts/patch_diskann_experience_core.py "$WORK/diskann"
"$PYC" scripts/patch_navhints_regional_history.py "$WORK/diskann"
git -C "$WORK/diskann" diff --check
git -C "$WORK/diskann" diff > "$ART/navhints-core.patch"
(
 cd "$WORK/diskann"
 nice -n 10 cargo +1.97.1 build --release --locked \
 -p diskann-benchmark --features disk-index > "$ART/build.log" 2>&1
)
BIN="$WORK/diskann/target/release/diskann-benchmark"
test -x "$BIN"
sha256sum "$BIN" > "$ART/search-binary-sha256.txt"
"$PYC" scripts/qualify_junction_score_ablation.py \
 --binary "$BIN" --dataset-manifest "$DATA_MANIFEST" \
 --index-prefix "$INDEX" --ivf-root "$ART/ivf" \
 --work "$WORK/evaluation" --out "$ART/results" --reps 3 \
 > "$ART/evaluation.log" 2>&1
test -s "$ART/results/junction-score-ablation.json"
echo "NAVHINTS_JUNCTION_SCORE_${DATASET}_PASS" > "$ART/status.txt"
