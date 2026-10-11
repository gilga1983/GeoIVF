#!/usr/bin/env bash
# PubMed full Core + Sample2 replay for coherent final-paper mechanism panels.
# Uses the SAME disjoint first-5K teacher trace, pinned graph and 4K/1K
# causal held-out sequence as the author-aligned Catapult head-to-head.
set -euxo pipefail

ART="$PWD/artifacts/final-core-starts"
WORK="$RUNNER_TEMP/final-core-starts-$GITHUB_RUN_ID"
ROOT="$HOME/.cache/geoivf"
REV="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
BASE="$ROOT/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
ALLQ="$ROOT/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$ROOT/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX="$ROOT/catapult-paper-pubmed1m-index-v1/diskann-index"
TRACE="$ROOT/high-l-medoid-teacher-traces-v1/medoid-teacher.L4.jsonl"
PORTALS="$ROOT/id-portal-pools-v1/portals-n512.bin"
STATE="$ROOT/navhints-fixed-memory-tradeoff-20261011"
HINTS="$STATE/landmarks/landmarks-b14464.bin"
IVF="$STATE/ivf/hints-b14464-nlist512-spherical.bin"

mkdir -p "$ART" "$WORK" "$STATE/landmarks" "$STATE/ivf"
for p in "$BASE" "$ALLQ" "$GT" "$TRACE" "$PORTALS"; do test -s "$p"; done
test -d "$(dirname "$INDEX")"
git rev-parse HEAD > "$ART/geoivf-commit.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
sha256sum "$BASE" "$ALLQ" "$GT" "$TRACE" "$PORTALS" > "$ART/input-sha256.txt"

python3 -m venv "$WORK/venv"
PYC="$WORK/venv/bin/python"
"$PYC" -m pip install numpy==2.2.6 > "$ART/pip.log" 2>&1
if [ ! -s "$HINTS" ] || [ ! -s "$HINTS.manifest.json" ]; then
  "$PYC" scripts/learn_global_navigation_landmarks.py \
    --trace "$TRACE" --portals "$PORTALS" \
    --out-dir "$STATE/landmarks" --budgets 14464 \
    > "$ART/learn.log" 2>&1
fi
if [ ! -s "$IVF" ] || [ ! -s "$IVF.manifest.json" ]; then
  "$PYC" scripts/build_hint_ivf.py \
    --base "$BASE" --hints "$HINTS" --out "$IVF" \
    --nlist 512 --iterations 5 --batch 2048 --seed 20261005 \
    > "$ART/build-ivf.log" 2>&1
fi
cp "$IVF.manifest.json" "$ART/ivf.manifest.json"
sha256sum "$IVF" "$HINTS" > "$ART/final-core-state-sha256.txt"
echo "FINAL_CORE: B=14464 C=2048 k=8 H=10 Sample2=1/2" > "$ART/controller-contract.txt"

# Construct a manifest of the *same* frozen last 5000 queries.
PYTHONPATH="$PWD/scripts" "$PYC" - "$ALLQ" "$GT" "$ART/final-pubmed-manifest.json" "$WORK/heldout5000.fbin" <<'PY'
import json,sys
from pathlib import Path
from qualify_navhints_catapult_headtohead import shape,slice_fbin
full,gt,manifest,dst=map(Path,sys.argv[1:])
if shape(full)!=(10000,768): raise ValueError("not frozen PubMed 10k 768d replay")
slice_fbin(full,dst,5000,5000)
manifest.write_text(json.dumps({
 "dataset":"PubMed1M/MedCPT",
 "split":{"static_train_rows":5000,"online_warmup_rows":4000,"measured_rows":1000},
 "data_type":"float32","metric":"inner_product",
 "files":{"heldout5000":str(dst.resolve()),"heldout5000_gt":str(gt.resolve())}
},indent=2)+"\n")
PY

export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
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
      > "$ART/build.log" 2>&1
)
BIN="$WORK/diskann/target/release/diskann-benchmark"
test -x "$BIN"
sha256sum "$BIN" > "$ART/search-binary-sha256.txt"

"$PYC" scripts/qualify_final_core_starts.py \
  --binary "$BIN" --heldout "$WORK/heldout5000.fbin" \
  --gt5000 "$GT" --index-prefix "$INDEX" --ivf "$IVF" \
  --work "$WORK/starts" --out "$ART/starts" > "$ART/starts.log" 2>&1
test -s "$ART/starts/final-core-starts.json"
echo "FINAL_CORE_STARTS_SUCCESS" | tee "$ART/status.txt"
