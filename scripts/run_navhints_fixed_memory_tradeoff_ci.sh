#!/usr/bin/env bash
# Strictly matched 103KB-routing-payload NavHints memory reallocation test.
# IMPORTANT: all evaluation is on a self-hosted runner; locks serialize SSD.
set -euxo pipefail
ART="$PWD/artifacts/navhints-fixed-memory-tradeoff"
WORK="$RUNNER_TEMP/navhints-fixed-memory-$GITHUB_RUN_ID"
ROOT="$HOME/.cache/geoivf"
LOCK="$ROOT/speed-device.lock"
REV="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"

BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT_HELD="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$ROOT/id-portal-pools-v1/portals-n512.bin"

STATE="$ROOT/paper-competitors-phase1-v1/navhints"
LANDMARK="$STATE/landmarks"
IVF="$STATE/hints-b16000-nlist512-spherical.bin"
mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")" "$LANDMARK"

for file in "$BASE" "$QUERIES" "$GT_HELD" "$TRACE" "$PORTALS"; do
  test -s "$file"
done
test -d "$(dirname "$INDEX")"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
printf '%s\n' "$REV" > "$ART/diskann-source-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
sha256sum "$BASE" "$QUERIES" "$GT_HELD" "$TRACE" "$PORTALS" > "$ART/input-hashes.txt"

# Serialize the whole experiment, including CPU-intensive source builds.
exec 9>"$LOCK"
echo "Waiting for exclusive CPU+SSD lock" | tee "$ART/lock.log"
flock 9
echo "Acquired lock $(date -Is)" | tee -a "$ART/lock.log"

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check "numpy==2.2.6" "faiss-cpu>=1.9,<2"
PYC="$WORK/venv/bin/python"

if [ ! -s "$LANDMARK/landmarks-b16000.bin" ]; then
  "$PYC" scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" --portals "$PORTALS" \
    --out-dir "$LANDMARK" --budgets "16000" > "$ART/build-landmarks.log" 2>&1
fi
if [ ! -s "$IVF" ] || [ ! -s "$IVF.manifest.json" ]; then
  "$PYC" scripts/build_hint_ivf.py --base "$BASE" \
    --hints "$LANDMARK/landmarks-b16000.bin" \
    --out "$IVF" --nlist 512 --iterations 5 --batch 2048 \
    --seed 20261005 > "$ART/build-ivf.log" 2>&1
fi
sha256sum "$IVF" "$IVF.manifest.json" "$LANDMARK/landmarks-b16000.bin" > "$ART/nav-routing-hashes.txt"
cp "$IVF.manifest.json" "$ART/nav-hint-ivf.manifest.json"


# A fixed-payload sweep replaces 1,536, 3,584, or 7,680 junction IDs
# with 2K, 4K, or 8K winner IDs. The 16K map is copied byte-for-byte
# from the already validated NavHints baseline.
SWEEP_IVF_DIR="$ROOT/navhints-fixed-memory-tradeoff-20261011/ivf"
SWEEP_LANDMARKS="$ROOT/navhints-fixed-memory-tradeoff-20261011/landmarks"
mkdir -p "$SWEEP_IVF_DIR" "$SWEEP_LANDMARKS"
cp "$IVF" "$SWEEP_IVF_DIR/hints-b16000-nlist512-spherical.bin"
cp "$IVF.manifest.json" "$SWEEP_IVF_DIR/hints-b16000-nlist512-spherical.bin.manifest.json"
cmp "$IVF" "$SWEEP_IVF_DIR/hints-b16000-nlist512-spherical.bin"
if [ ! -s "$SWEEP_LANDMARKS/landmarks-b14464.bin" ]; then
  "$PYC" scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" --portals "$PORTALS" \
    --out-dir "$SWEEP_LANDMARKS" --budgets "8320,12416,14464,15488" \
    > "$ART/sweep-learn-junctions.log" 2>&1
fi
for B in 8320 12416 14464 15488; do
  FILE="$SWEEP_IVF_DIR/hints-b${B}-nlist512-spherical.bin"
  if [ ! -s "$FILE" ] || [ ! -s "$FILE.manifest.json" ]; then
    "$PYC" scripts/build_hint_ivf.py \
      --base "$BASE" --hints "$SWEEP_LANDMARKS/landmarks-b${B}.bin" \
      --out "$FILE" --nlist 512 --iterations 5 --batch 2048 \
      --seed 20261005 > "$ART/cluster-b${B}.log" 2>&1
  fi
done
sha256sum "$SWEEP_IVF_DIR"/hints-b*-nlist512-spherical.bin \
    > "$ART/ivf-sweep-hashes.txt"
cp "$SWEEP_IVF_DIR"/hints-b*-nlist512-spherical.bin.manifest.json "$ART/"

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
fi
rustup toolchain install 1.97.1 --profile minimal

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/navhints"
git -C "$WORK/navhints" checkout --detach "$REV"
test "$(git -C "$WORK/navhints" rev-parse HEAD)" = "$REV"
"$PYC" scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/navhints"
"$PYC" scripts/patch_diskann_experience_core.py "$WORK/navhints"
"$PYC" scripts/patch_navhints_regional_history.py "$WORK/navhints"
git -C "$WORK/navhints" diff --check
git -C "$WORK/navhints" diff > "$ART/navhints-core.patch"
(
  cd "$WORK/navhints"
  cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
    > "$ART/navhints-build.log" 2>&1
)
NAV_BIN="$WORK/navhints/target/release/diskann-benchmark"
test -x "$NAV_BIN"
sha256sum "$NAV_BIN" > "$ART/binaries-sha256.txt"

"$PYC" scripts/qualify_navhints_fixed_memory_tradeoff.py \
  --nav-bin "$NAV_BIN" --queries "$QUERIES" \
  --heldout-gt "$GT_HELD" \
  --index-prefix "$INDEX" --ivf-dir "$SWEEP_IVF_DIR" \
  --work "$WORK/replay" --out "$ART/results" \
  > "$ART/evaluation.log" 2>&1
grep "NAVHINTS_FIXED_PAYLOAD_TRADEOFF_PASS" "$ART/evaluation.log"
echo "NAVHINTS_FIXED_PAYLOAD_TRADEOFF_VALIDATED" | tee "$ART/status.txt"
