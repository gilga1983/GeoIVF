#!/usr/bin/env bash
# Author-source oracle + pinned DiskANN compile on SELF-HOSTED runner.
# Pure fidelity verification, no ANN data downloads or benchmark outcomes.
set -euxo pipefail
ART="$PWD/artifacts/catapult-fidelity-audit"
WORK="$RUNNER_TEMP/catapult-fidelity-audit-$GITHUB_RUN_ID"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
AUTHOR_SHA="a16eddd34b4339db5ec86e292470ce7929179bc3"
DISKANN_SHA="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")"
git rev-parse HEAD > "$ART/geoivf-audit-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
exec 9>"$LOCK"
echo "Waiting for exclusive self-hosted CPU+SSD lock" | tee "$ART/lock.log"
flock 9
echo "Acquired lock $(date -Is)" | tee -a "$ART/lock.log"

git clone --filter=blob:none https://github.com/MRandl/catapult-db.git "$WORK/author"
git -C "$WORK/author" checkout --detach "$AUTHOR_SHA"
test "$(git -C "$WORK/author" rev-parse HEAD)" = "$AUTHOR_SHA"
printf '%s\n' "$AUTHOR_SHA" > "$ART/original-author-source-sha.txt"

# Verify expected discovered differences directly against original code.
python3 scripts/audit_catapult_source_parity.py \
 --author "$WORK/author" \
 --port scripts/patch_diskann_paper_catapult.py --out "$ART/source-only"

export RUSTUP_HOME="$HOME/.cache/geoivf/catapult-fidelity-rustup"
export CARGO_HOME="$HOME/.cache/geoivf/catapult-fidelity-cargo"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/install-rustup.sh"
  sh "$WORK/install-rustup.sh" -y --profile minimal --no-modify-path --default-toolchain nightly-2026-09-15
fi
rustup toolchain install nightly-2026-09-15 --profile minimal
cp scripts/catapult_native_fidelity_oracle.rs "$WORK/author/src/bin/navhints_fidelity_oracle.rs"
(
  cd "$WORK/author"
  cargo +nightly-2026-09-15 run --release --locked --bin navhints_fidelity_oracle \
    >"$ART/native-oracle.log" 2>&1
)
grep 'ORIGINAL_CATAPULT_NATIVE_ORACLE_PASS' "$ART/native-oracle.log"
git -C "$WORK/author" diff --exit-code  # original tracked author files unchanged

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/diskann"
git -C "$WORK/diskann" checkout --detach "$DISKANN_SHA"
test "$(git -C "$WORK/diskann" rev-parse HEAD)" = "$DISKANN_SHA"
python3 scripts/patch_diskann_paper_catapult.py "$WORK/diskann"
# The author alignment is an ISOLATED optional verification variant.
python3 scripts/align_catapult_author_semantics.py "$WORK/diskann"
python3 scripts/audit_catapult_source_parity.py \
 --author "$WORK/author" \
 --port scripts/patch_diskann_paper_catapult.py \
 --patched "$WORK/diskann" --out "$ART/source-and-aligned"

git -C "$WORK/diskann" diff --check
git -C "$WORK/diskann" diff > "$ART/diskann-author-aligned-port.patch"
test -s "$ART/diskann-author-aligned-port.patch"

# Execute the actual compiler, not just a string-based source audit.
export RUSTFLAGS="-C target-cpu=native"
(
 cd "$WORK/diskann"
 cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
    >"$ART/diskann-aligned-build.log" 2>&1
)
grep 'ORIGINAL_CATAPULT_NATIVE_ORACLE_PASS' "$ART/native-oracle.log"
test -x "$WORK/diskann/target/release/diskann-benchmark"
printf '%s\n' "CATAPULT_ORIGINAL_AUTHOR_FIDELITY_AUDIT_AND_COMPILE_PASS" | tee "$ART/success.txt"
