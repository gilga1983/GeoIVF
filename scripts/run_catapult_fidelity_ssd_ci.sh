#!/usr/bin/env bash
# Pinned same-graph SSD sensitivity for ORIGINAL vs author-aligned Catapult port.
# All variants use native Microsoft DiskANN3, not author's in-memory engine.
set -euxo pipefail
ART="$PWD/artifacts/catapult-fidelity-ssd"
WORK="$RUNNER_TEMP/catapult-fidelity-ssd-$GITHUB_RUN_ID"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
REV="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
ROOT="$HOME/.cache/geoivf"
QUERIES="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
TRAIN_GT="$ROOT/catapult-paper-pubmed1m/throughput-only.gt"
EVAL_GT="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")"
for f in "$QUERIES" "$TRAIN_GT" "$EVAL_GT"; do test -s "$f"; done
test -s "$INDEX"
git rev-parse HEAD > "$ART/geoivf-audit-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
sha256sum "$QUERIES" "$TRAIN_GT" "$EVAL_GT" > "$ART/input-hashes.txt"

exec 9>"$LOCK"
echo "Waiting for single-machine experiment lock" | tee "$ART/lock.log"
flock 9
echo "Acquired exclusive lock $(date -Is)" | tee -a "$ART/lock.log"

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/diskann"
git -C "$WORK/diskann" checkout --detach "$REV"
test "$(git -C "$WORK/diskann" rev-parse HEAD)" = "$REV"
printf '%s\n' "$REV" > "$ART/original-DiskANN-source-sha.txt"

export CARGO_HOME="$HOME/.cache/geoivf/catapult-fidelity-cargo"
export RUSTUP_HOME="$HOME/.cache/geoivf/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
fi
rustup toolchain install 1.97.1 --profile minimal

# Compile legacy reference source and keep its binary, then adjust only the
# three author-semantics features plus latency timing in the same checkout.
"$WORK/venv/bin/python" scripts/patch_diskann_catapult_snapshot.py "$WORK/diskann"
(
 cd "$WORK/diskann"
 cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
   >"$ART/build-legacy.log" 2>&1
)
cp "$WORK/diskann/target/release/diskann-benchmark" "$WORK/catapult-legacy"
git -C "$WORK/diskann" diff --check
git -C "$WORK/diskann" diff > "$ART/port-legacy.patch"

"$WORK/venv/bin/python" scripts/align_catapult_author_semantics.py "$WORK/diskann"
(
 cd "$WORK/diskann"
 cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
   >"$ART/build-author-aligned.log" 2>&1
)
cp "$WORK/diskann/target/release/diskann-benchmark" "$WORK/catapult-author-aligned"
git -C "$WORK/diskann" diff --check
git -C "$WORK/diskann" diff > "$ART/port-author-aligned.patch"
sha256sum "$WORK/catapult-legacy" "$WORK/catapult-author-aligned" > "$ART/binary-sha256.txt"
test -x "$WORK/catapult-legacy"
test -x "$WORK/catapult-author-aligned"
test -s "$ART/port-author-aligned.patch"

# The Python runner trains strictly chronologically using ONE worker, freezes
# the trained bucket tables, then evaluates the SAME 5K suffix at 4 workers.
"$WORK/venv/bin/python" scripts/qualify_catapult_fidelity_ssd.py \
 --legacy-bin "$WORK/catapult-legacy" --author-bin "$WORK/catapult-author-aligned" \
 --queries "$QUERIES" --train-gt "$TRAIN_GT" --eval-gt "$EVAL_GT" \
 --index-prefix "$INDEX" --work "$WORK/catapult-states" \
 --out "$ART/results" >"$ART/qualification.log" 2>&1
grep 'CATAPULT_FIDELITY_SSD_3SEED_MATCHED_L_COMPLETED' "$ART/qualification.log"
echo "CATAPULT_SAME_SSD_AUTHOR_ALIGNMENT_VALIDATED" | tee "$ART/result.txt"
