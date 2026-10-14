#!/usr/bin/env bash
# Score-only NavHints controls for Text2Image-10M and Coveo on the original
# real-SSD index and frozen teacher trace. No external competitor reruns.
set -euo pipefail
DATASET="${1:?usage: run_junction_score_remaining_ci.sh text2image|coveo}"
ROOT="$HOME/.cache/geoivf"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/junction-score-remaining/$DATASET"
WORK="${RUNNER_TEMP:-/tmp}/junction-score-$DATASET-${GITHUB_RUN_ID:-manual}"
mkdir -p "$ART" "$WORK" "$ART/selection" "$ART/ivf" "$ART/results"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"

case "$DATASET" in
text2image)
  CACHE="$ROOT/public-eval/text2image-10M"
  DATA_MANIFEST="$CACHE/data/dataset.manifest.json"
  INDEX="$CACHE/index/diskann-index"
  TRACE="$CACHE/trace/teacher.jsonl"
  STATE="$CACHE/navhints"
  REFERENCE="$STATE/landmarks-b14464.bin"
  REF_IVF="$STATE/hints-b14464-nlist512.bin"
  PARTITION=spherical
  DATA_TYPE=float32
  B=14464
  ;;
coveo)
  CACHE="$ROOT/coveo-real-demand-v1"
  DATA_MANIFEST="$CACHE/split/dataset.manifest.json"
  REPLAY_GT="$CACHE/split/replay.gt"
  INDEX="$CACHE/index/diskann-index"
  TRACE="$CACHE/trace/teacher.jsonl"
  STATE="$CACHE/navhints-final-core-20261011"
  REFERENCE="$STATE/landmarks-b14464.bin"
  REF_IVF="$STATE/hints-b14464-nlist512.bin"
  PARTITION=spherical
  DATA_TYPE=float32
  B=8192 # explicitly rank-sensitive: selected Coveo B=12169 retains all IDs
  ;;
*) echo "Unsupported dataset $DATASET" >&2; exit 2;;
esac
for path in "$DATA_MANIFEST" "$TRACE" "$REFERENCE" "$REF_IVF" "$REF_IVF.manifest.json"; do
  test -s "$path" || { echo "MISSING_FROZEN_INPUT: $path" >&2; exit 10; }
done
test -s "$INDEX.complete.json" || { echo "MISSING_FROZEN_INDEX: $INDEX" >&2; exit 11; }
if [ "$DATASET" = coveo ]; then test -s "$REPLAY_GT"; fi

BASE="$(python3 - "$DATA_MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d["files"]["base"])
PY
)"
# Coveo manifest's base path and public dataset's base path are both
# canonical frozen embeddings, not reconstructed or approximate copies.
test -s "$BASE"
sha256sum "$TRACE" "$REFERENCE" "$REF_IVF" "$BASE" > "$ART/frozen-input-sha256.txt"
cp "$DATA_MANIFEST" "$ART/input-dataset-manifest.json"

python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install numpy==2.2.6 > "$ART/pip.log" 2>&1
"$PYC" scripts/learn_junction_score_variants.py \
 --trace "$TRACE" --reference-sum "$REFERENCE" --budget "$B" \
 --out "$ART/selection" > "$ART/selection/learn.log" 2>&1
test -s "$ART/selection/selection.json"

if [ "$DATASET" = coveo ]; then
  # Exact original-Coveo geometry check at its SELECTED B=12169 before
  # evaluating the separate B8192 ranking-sensitive experiment.
  # The original 2026-10-11 Coveo builder used numpy==2.2.6.
  "$PYC" scripts/build_hint_ivf_generic.py \
    --base "$BASE" --base-data-type float32 --hints "$REFERENCE" \
    --out "$ART/ivf/original-selected-coveo.bin" \
    --partition spherical --nlist 512 --iterations 5 --batch 1024 \
    --seed 20261005 > "$ART/ivf/original-selected-coveo.log" 2>&1
  cmp "$REF_IVF" "$ART/ivf/original-selected-coveo.bin"
  echo "ORIGINAL_FULL_COVEO_DIRECTORY_BYTE_IDENTICAL" > "$ART/repro.status"
else
  # Public 10M routing states were built with system Python/NumPy.
  # Verify the original after reconstructing identical top-14464 selection.
  echo "ORIGINAL_T2I_DIRECTORY_REQUIRES_EXACT_COMPARISON" > "$ART/repro.status"
fi

for mode in sum count mean; do
  HINTS="$ART/selection/landmarks-b$B-$mode.bin"
  IVF="$ART/ivf/ivf-$mode.bin"
  if [ "$DATASET" = coveo ]; then
    PY="$PYC"
  else
    PY=python3
  fi
  "$PY" scripts/build_hint_ivf_generic.py \
     --base "$BASE" --base-data-type "$DATA_TYPE" --hints "$HINTS" \
     --out "$IVF" --partition "$PARTITION" \
     --nlist 512 --iterations 5 --batch 1024 --seed 20261005 \
     > "$ART/ivf/$mode.log" 2>&1
  test -s "$IVF" && test -s "$IVF.manifest.json"
  sha256sum "$HINTS" "$IVF" >> "$ART/variant-sha256.txt"
done

# Never move past routing-state reproducibility checks on a mismatch.
if [ "$DATASET" = text2image ]; then
  cmp "$REF_IVF" "$ART/ivf/ivf-sum.bin"
  echo "ORIGINAL_T2I_DIRECTORY_BYTE_IDENTICAL" > "$ART/repro.status"
else
  # For Coveo the original 12,169-ID directory has been checked above.
  # The B8192 sum-ranked ID list is exactly its frozen first 8192 entries
  # (checked by learner); variants differ only in B8192 junction selection.
  test "$B" -eq 8192
fi

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
command -v rustup
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

if [ "$DATASET" = coveo ]; then
  "$PYC" scripts/qualify_junction_score_coveo.py \
    --binary "$BIN" --dataset-manifest "$DATA_MANIFEST" --replay-gt "$REPLAY_GT" \
    --index-prefix "$INDEX" --ivf-root "$ART/ivf" \
    --work "$WORK/eval" --out "$ART/results" --reps 3 \
    > "$ART/evaluation.log" 2>&1
else
  "$PYC" scripts/qualify_junction_score_ablation.py \
    --binary "$BIN" --dataset-manifest "$DATA_MANIFEST" \
    --index-prefix "$INDEX" --ivf-root "$ART/ivf" \
    --work "$WORK/eval" --out "$ART/results" --reps 3 \
    > "$ART/evaluation.log" 2>&1
fi
test -s "$ART/results/junction-score-ablation.json"
echo "NAVHINTS_SCORE_${DATASET}_PASS" > "$ART/status.txt"
