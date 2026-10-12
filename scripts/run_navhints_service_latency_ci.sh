#!/usr/bin/env bash
# Freeze one SSD graph and query stream; measure complete causal request scopes.
# Single-machine, single-device experiment: the lock covers builds and replays.
set -euxo pipefail

ART="$PWD/artifacts/navhints-complete-service-pubmed"
ROOT="$HOME/.cache/geoivf"
WORK="$RUNNER_TEMP/navhints-complete-service-$GITHUB_RUN_ID"
LOCK="$ROOT/speed-device.lock"
REV="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
AUTHOR="a16eddd34b4339db5ec86e292470ce7929179bc3"

BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
ALLQ="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT_TRAIN="$ROOT/catapult-paper-pubmed1m/throughput-only.gt"
GT_HELD="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$ROOT/id-portal-pools-v1/portals-n512.bin"
STATE="$ROOT/navhints-fixed-memory-tradeoff-20261011"
HINTS="$STATE/landmarks/landmarks-b14464.bin"
IVF="$STATE/ivf/hints-b14464-nlist512-spherical.bin"

mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")" "$STATE/landmarks" "$STATE/ivf"
for path in "$BASE" "$ALLQ" "$GT_TRAIN" "$GT_HELD" "$TRACE" "$PORTALS"; do test -s "$path"; done
test -d "$(dirname "$INDEX")"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
printf '%s\n' "$REV" > "$ART/diskann-pinned-commit.txt"
printf '%s\n' "$AUTHOR" > "$ART/catapult-author-commit.txt"
sha256sum "$BASE" "$ALLQ" "$GT_TRAIN" "$GT_HELD" "$TRACE" "$PORTALS" > "$ART/inputs-sha256.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; lsblk; } > "$ART/host.txt"

exec 9>"$LOCK"
echo "Waiting for shared SSD/CPU experiment lock" | tee "$ART/lock.log"
flock 9
echo "Acquired lock $(date -Is)" | tee -a "$ART/lock.log"

python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install --disable-pip-version-check "numpy==2.2.6" "faiss-cpu>=1.9,<2" > "$ART/pip.log" 2>&1

if [ ! -s "$HINTS" ]; then
  "$PYC" scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" --portals "$PORTALS" --out-dir "$STATE/landmarks" \
    --budgets 14464 > "$ART/learn.log" 2>&1
fi
if [ ! -s "$IVF" ] || [ ! -s "$IVF.manifest.json" ]; then
  "$PYC" scripts/build_hint_ivf.py \
    --base "$BASE" --hints "$HINTS" --out "$IVF" \
    --nlist 512 --iterations 5 --batch 2048 --seed 20261005 \
    > "$ART/ivf-build.log" 2>&1
fi
sha256sum "$IVF" "$IVF.manifest.json" "$HINTS" > "$ART/routing-sha256.txt"
cp "$IVF.manifest.json" "$ART/routing.manifest.json"

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
fi
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/navhints"
git -C "$WORK/navhints" checkout --detach "$REV"
test "$(git -C "$WORK/navhints" rev-parse HEAD)" = "$REV"
"$PYC" scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/navhints"
"$PYC" scripts/patch_diskann_experience_core.py "$WORK/navhints"
"$PYC" scripts/patch_navhints_regional_history.py "$WORK/navhints"
"$PYC" scripts/patch_diskann_full_service.py "$WORK/navhints" --engine nav
git -C "$WORK/navhints" diff --check
git -C "$WORK/navhints" diff > "$ART/navhints-service.patch"
(
  cd "$WORK/navhints"
  cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
    > "$ART/navhints-build.log" 2>&1
)
NAV_BIN="$WORK/navhints/target/release/diskann-benchmark"
test -x "$NAV_BIN"

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/catapult"
git -C "$WORK/catapult" checkout --detach "$REV"
test "$(git -C "$WORK/catapult" rev-parse HEAD)" = "$REV"
"$PYC" scripts/patch_diskann_catapult_snapshot.py "$WORK/catapult"
"$PYC" scripts/align_catapult_author_semantics.py "$WORK/catapult"
"$PYC" scripts/patch_catapult_author_causal_eval.py "$WORK/catapult"
"$PYC" scripts/patch_diskann_full_service.py "$WORK/catapult" --engine catapult
git -C "$WORK/catapult" diff --check
git -C "$WORK/catapult" diff > "$ART/catapult-service.patch"
(
  cd "$WORK/catapult"
  cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
    > "$ART/catapult-build.log" 2>&1
)
CAT_BIN="$WORK/catapult/target/release/diskann-benchmark"
test -x "$CAT_BIN"
sha256sum "$NAV_BIN" "$CAT_BIN" > "$ART/binaries-sha256.txt"

"$PYC" scripts/qualify_navhints_service_latency.py \
  --nav-bin "$NAV_BIN" --cat-bin "$CAT_BIN" \
  --queries "$ALLQ" --train-gt "$GT_TRAIN" --heldout-gt "$GT_HELD" \
  --index-prefix "$INDEX" --ivf "$IVF" \
  --work "$WORK/replay" --out "$ART/results" \
  --widths "${SERVICE_WIDTHS:-40,80,160}" \
  --reps "${SERVICE_REPS:-1}" --cat-seeds "${SERVICE_CAT_SEEDS:-0}" \
  > "$ART/evaluation.log" 2>&1
test -s "$ART/results/complete-service-pubmed.json"
grep -F "COMPLETE_SERVICE_CAUSAL_PASS" "$ART/evaluation.log"
echo "COMPLETE_SERVICE_PUBMED_PASS" | tee "$ART/status.txt"

# The GitHub Action's cleanup step removes only its unique scratch checkout.
