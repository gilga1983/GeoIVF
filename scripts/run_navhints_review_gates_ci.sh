#!/usr/bin/env bash
# Paired reviewer controls: score selection and equal-payload k=8 allocations.
# Runs ONLY on the geoivf self-hosted machine; lock protects SSD and CPU.
set -euxo pipefail

ROOT="$HOME/.cache/geoivf"
REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/navhints-review-gates-20261013"
WORK="$RUNNER_TEMP/navhints-review-gates-$GITHUB_RUN_ID"
BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
ALLQ="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$ROOT/id-portal-pools-v1/portals-n512.bin"

# Keep all new state and patched binaries isolated from the frozen cache.
LANDMARK="$WORK/learned"
ALLOC="$WORK/alloc_ivf"
SCORE_IVF="$WORK/score_ivf"
mkdir -p "$ROOT" "$ART" "$WORK" "$LANDMARK" "$ALLOC" "$SCORE_IVF"

for p in "$BASE" "$ALLQ" "$GT" "$TRACE" "$PORTALS"; do test -s "$p"; done
test -s "$INDEX.complete.json" || test -d "$(dirname "$INDEX")"
git rev-parse HEAD > "$ART/geoivf-source-sha.txt"
printf '%s\n' "$REV" > "$ART/diskann-source-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
sha256sum "$BASE" "$ALLQ" "$GT" "$TRACE" "$PORTALS" > "$ART/input-sha256.txt"

exec 9>"$ROOT/speed-device.lock"
echo "Waiting for exclusive real-SSD/CPU lock" > "$ART/lock.txt"
flock 9
date -Is >> "$ART/lock.txt"

python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install --disable-pip-version-check "numpy==2.2.6" "faiss-cpu>=1.9,<2" \
    > "$ART/pip.log" 2>&1

# Important validation: the control learner MUST emit the identical
# position-sum landmark IDs used by the current paper.
"$PY" scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" --portals "$PORTALS" \
    --out-dir "$LANDMARK/published" \
    --budgets "8320,12416,14464,15488" \
    > "$ART/learn-reference.log" 2>&1

"$PY" scripts/prepare_navhints_score_controls.py \
    --trace "$TRACE" --out-dir "$LANDMARK/scores" --budget 14464 \
    --random-seed 20261013 \
    --published-landmarks "$LANDMARK/published/landmarks-b14464.bin" \
    > "$ART/learn-score-controls.log" 2>&1
cp "$LANDMARK/scores/score-controls.manifest.json" "$ART/"

# Same 512-region clustering and random seed for all 3 scoring arms,
# and the same total 102,940-byte routing payload for all allocations.
for B in 8320 12416 14464 15488; do
    "$PY" scripts/build_hint_ivf.py \
        --base "$BASE" \
        --hints "$LANDMARK/published/landmarks-b$B.bin" \
        --out "$ALLOC/hints-b$B-nlist512-spherical.bin" \
        --nlist 512 --iterations 5 --batch 2048 --seed 20261005 \
        > "$ART/build-allocation-b$B.log" 2>&1
done

for MODE in position-sum frequency random; do
    mkdir -p "$SCORE_IVF/$MODE"
    "$PY" scripts/build_hint_ivf.py \
        --base "$BASE" \
        --hints "$LANDMARK/scores/$MODE/landmarks-b14464.bin" \
        --out "$SCORE_IVF/$MODE/hints-b14464-nlist512-spherical.bin" \
        --nlist 512 --iterations 5 --batch 2048 --seed 20261005 \
        > "$ART/build-score-$MODE.log" 2>&1
done

cmp "$SCORE_IVF/position-sum/hints-b14464-nlist512-spherical.bin" \
    "$ALLOC/hints-b14464-nlist512-spherical.bin"
sha256sum "$ALLOC"/*.bin "$SCORE_IVF"/*/*.bin > "$ART/ivf-sha256.txt"
cp "$ALLOC"/*.manifest.json "$ART/"
for MODE in position-sum frequency random; do
    cp "$SCORE_IVF/$MODE/hints-b14464-nlist512-spherical.bin.manifest.json" \
        "$ART/score-$MODE-ivf.manifest.json"
done

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
if ! command -v rustup >/dev/null 2>&1; then
    curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
    sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
fi
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN"
git -C "$WORK/DiskANN" checkout --detach "$REV"
test "$(git -C "$WORK/DiskANN" rev-parse HEAD)" = "$REV"
"$PY" scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/DiskANN"
"$PY" scripts/patch_diskann_experience_core.py "$WORK/DiskANN"
"$PY" scripts/patch_navhints_regional_history.py "$WORK/DiskANN"
git -C "$WORK/DiskANN" diff --check
git -C "$WORK/DiskANN" diff > "$ART/review-gates-diskann.patch"
(
 cd "$WORK/DiskANN"
 cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
     > "$ART/build-diskann.log" 2>&1
)
BIN="$WORK/DiskANN/target/release/diskann-benchmark"
test -x "$BIN"
sha256sum "$BIN" > "$ART/search-binary-sha256.txt"

"$PY" scripts/qualify_navhints_review_gates.py \
    --nav-bin "$BIN" --queries "$ALLQ" \
    --heldout-gt "$GT" --index-prefix "$INDEX" \
    --score-ivf-dir "$SCORE_IVF" --allocation-ivf-dir "$ALLOC" \
    --work "$WORK/replay" --out "$ART/results" --repetitions 3 \
    > "$ART/evaluation.log" 2>&1
grep -F "NAVHINTS_REVIEW_GATES_1_AND_2_PASS" "$ART/evaluation.log"
printf 'REVIEW_GATES_1_AND_2_VALIDATED\n' > "$ART/status.txt"
