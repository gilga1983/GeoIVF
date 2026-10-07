#!/usr/bin/env bash
set -euxo pipefail

DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5
ART="$PWD/artifacts/final-locality"
mkdir -p "$ART"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6

HELDOUT="$RUNNER_TEMP/final-locality-heldout-$GITHUB_RUN_ID.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT5000="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
IVF="$HOME/.cache/geoivf/paper-competitors-phase1-v1/navhints/hints-b16000-nlist512-spherical.bin"
for f in "$QUERIES" "$GT5000" "$IVF" "$INDEX"_disk.index "$INDEX"_pq_pivots.bin "$INDEX"_pq_compressed.bin; do test -s "$f"; done

.venv/bin/python - "$QUERIES" "$HELDOUT" <<'PY'
import struct,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
with src.open("rb") as f:
    rows,dim=struct.unpack("<II",f.read(8)); assert rows==10000
    f.seek(8+5000*dim*4); body=f.read(5000*dim*4)
with dst.open("wb") as f:
    f.write(struct.pack("<II",5000,dim)); f.write(body)
PY

export CARGO_HOME="$RUNNER_TEMP/finalloc-cargo-$GITHUB_RUN_ID"
export RUSTUP_HOME="$RUNNER_TEMP/finalloc-rustup-$GITHUB_RUN_ID"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/finalloc-rustup-$GITHUB_RUN_ID.sh"
sh "$RUNNER_TEMP/finalloc-rustup-$GITHUB_RUN_ID.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

WORK="$RUNNER_TEMP/final-locality-$GITHUB_RUN_ID"
rm -rf "$WORK"; mkdir -p "$WORK"
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/DiskANN"
git -C "$WORK/DiskANN" checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/DiskANN"
.venv/bin/python scripts/patch_diskann_experience_core.py "$WORK/DiskANN"
(cd "$WORK/DiskANN" && cargo fmt --all && git diff --check && cargo build --release --locked -p diskann-benchmark --features disk-index) 2>&1 | tee "$ART/build.log"
BIN="$WORK/DiskANN/target/release/diskann-benchmark"

mkdir -p "$ART/results"
.venv/bin/python scripts/qualify_final_locality.py   --binary "$BIN" --heldout "$HELDOUT" --gt5000 "$GT5000"   --index-prefix "$INDEX" --ivf "$IVF" --work "$WORK/eval" --out "$ART/results"   2>&1 | tee "$ART/eval.log"

rm -rf "$WORK" "$HELDOUT"
