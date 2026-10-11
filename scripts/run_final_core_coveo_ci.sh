#!/usr/bin/env bash
# Reproduce final Core on the SAME frozen licensed chronological Coveo split.
# Never re-download private source data, reset the old directory, or rebuild
# the published SSD graph. Runs ONLY on self-hosted geoivf runners.
set -euxo pipefail
ROOT="$HOME/.cache/geoivf/coveo-real-demand-v1"
BASE="$ROOT/prepared/coveo-products-cosine.fbin"
SPLIT="$ROOT/split"
TRACE="$ROOT/trace/teacher.jsonl"
INDEX_PREFIX="$ROOT/index/diskann-index"
STATE="$ROOT/navhints-final-core-20261011"
LANDMARKS="$STATE/landmarks-b14464.bin"
IVF="$STATE/hints-b14464-nlist512.bin"
ART="$PWD/artifacts/final-core-coveo"
WORK="$RUNNER_TEMP/final-core-coveo-$GITHUB_RUN_ID"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

mkdir -p "$ART" "$WORK" "$STATE"
for f in "$BASE" "$SPLIT/dataset.manifest.json" "$SPLIT/replay.gt" \
         "$TRACE" "$INDEX_PREFIX.complete.json"; do test -s "$f"; done

git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
sha256sum "$BASE" "$SPLIT/dataset.manifest.json" "$SPLIT/replay.gt" "$TRACE" \
    "$INDEX_PREFIX.complete.json" > "$ART/frozen-input-sha256.txt"

python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install numpy==2.2.6 > "$ART/python-packages.log" 2>&1

if [ ! -s "$LANDMARKS" ] || [ ! -s "$LANDMARKS.manifest.json" ]; then
  "$PYC" scripts/learn_navhints_generic.py \
    --trace "$TRACE" --out "$LANDMARKS" --budget 14464 \
    > "$ART/learn.log" 2>&1
fi
if [ ! -s "$IVF" ] || [ ! -s "$IVF.manifest.json" ]; then
  "$PYC" scripts/build_hint_ivf_generic.py \
    --base "$BASE" --base-data-type float32 --hints "$LANDMARKS" \
    --out "$IVF" --partition spherical --nlist 512 \
    --iterations 5 --batch 1024 --seed 20261005 \
    > "$ART/build-ivf.log" 2>&1
fi
cp "$LANDMARKS.manifest.json" "$ART/landmarks.manifest.json"
cp "$IVF.manifest.json" "$ART/ivf.manifest.json"
sha256sum "$LANDMARKS" "$IVF" > "$ART/final-core-state-sha256.txt"
echo "FINAL_CORE: B<=14464 (Coveo eligibility capped); C=2048; k=8; Full Sample2=1/2 H=10" > "$ART/controller-contract.txt"

export CARGO_HOME="$HOME/.cache/geoivf/catapult-fidelity-cargo"
export RUSTUP_HOME="$HOME/.cache/geoivf/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
command -v rustup
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/diskann" \
  > "$ART/clone.log" 2>&1
git -C "$WORK/diskann" checkout --detach "$REV"
test "$(git -C "$WORK/diskann" rev-parse HEAD)" = "$REV"
"$PYC" scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/diskann"
"$PYC" scripts/patch_diskann_experience_core.py "$WORK/diskann"
"$PYC" scripts/patch_navhints_regional_history.py "$WORK/diskann"
git -C "$WORK/diskann" diff --check
git -C "$WORK/diskann" diff > "$ART/final-core.patch"
(
  cd "$WORK/diskann"
  cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
    > "$ART/build-diskann.log" 2>&1
)
BIN="$WORK/diskann/target/release/diskann-benchmark"
test -x "$BIN"
sha256sum "$BIN" > "$ART/binary-sha256.txt"

"$PYC" scripts/qualify_final_core_coveo.py \
  --binary "$BIN" --dataset-manifest "$SPLIT/dataset.manifest.json" \
  --replay-gt "$SPLIT/replay.gt" --index-prefix "$INDEX_PREFIX" \
  --ivf "$IVF" --work "$WORK/eval" --out "$ART/results" --reps 3 \
  > "$ART/evaluation.log" 2>&1
test -s "$ART/results/coveo-final-core.json"
echo "FINAL_CORE_COVEO_CAUSAL_PASS" | tee "$ART/status.txt"
