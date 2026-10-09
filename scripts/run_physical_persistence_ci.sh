#!/usr/bin/env bash
# Self-hosted only: write 4KiB sectors exclusively on *separate* PubMed graph
# copies. Original index/data and approved NavHints paper are read-only.
set -euxo pipefail
DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/navhints-physical-persistence"
WORK="$RUNNER_TEMP/navhints-physical-persistence-$GITHUB_RUN_ID"
mkdir -p "$ART" "$WORK" "$ART/results"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
HELD_GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
IVF="$HOME/.cache/geoivf/paper-competitors-phase1-v1/navhints/hints-b16000-nlist512-spherical.bin"
for f in "$QUERIES" "$HELD_GT" "$INDEX_PREFIX""_disk.index" \
         "$INDEX_PREFIX""_pq_pivots.bin" "$INDEX_PREFIX""_pq_compressed.bin" \
         "$IVF" "$IVF.manifest.json"; do test -s "$f"; done
git rev-parse HEAD >"$ART/geoivf-commit.txt"
{ uname -a; lscpu; free -h; df -h; } >"$ART/host.txt"
echo "$DISKANN_REV" >"$ART/diskann-revision.txt"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6
"$WORK/venv/bin/python" -m py_compile scripts/patch_diskann_experience_physical.py \
  scripts/qualify_physical_persistence.py
export CARGO_HOME="$WORK/cargo"
export RUSTUP_HOME="$WORK/rustup"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-install.sh"
sh "$WORK/rustup-install.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN"
git -C "$WORK/DiskANN" checkout --detach "$DISKANN_REV"
python3 scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/DiskANN"
python3 scripts/patch_diskann_experience_core.py "$WORK/DiskANN"
python3 scripts/patch_diskann_experience_physical.py "$WORK/DiskANN"
git -C "$WORK/DiskANN" diff --check
git -C "$WORK/DiskANN" diff >"$ART/physical-index-experiment.diff"
(cd "$WORK/DiskANN" && cargo fmt --all && \
 cargo build --release --locked -p diskann-benchmark --features disk-index) \
    >"$ART/build.log" 2>&1
BIN="$WORK/DiskANN/target/release/diskann-benchmark"
test -x "$BIN"

"$WORK/venv/bin/python" scripts/qualify_physical_persistence.py \
  --binary "$BIN" --queries "$QUERIES" --groundtruth "$HELD_GT" \
  --index-prefix "$INDEX_PREFIX" --ivf "$IVF" \
  --work "$WORK/copied-index-tests" --out "$ART/results" --reps 3 \
  >"$ART/evaluation.log" 2>&1
test -s "$ART/results/physical-persistence.json"
echo "ISOLATED_PHYSICAL_PAGE_COST_SUCCESS"
