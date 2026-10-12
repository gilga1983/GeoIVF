#!/usr/bin/env bash
# End-to-end causal search+feedback latency on ONE frozen dataset and SSD graph.
# No index building, downloading, or changing frozen competitors/algorithms.
# The HOME-level lock serializes all phases on the measured device.
set -euxo pipefail

DATASET="${1:?usage: run_navhints_service_multidata_ci.sh coveo|text2image-10M|bigann-10M}"
REPO="$PWD"
ROOT="$HOME/.cache/geoivf"
ART="$PWD/artifacts/navhints-complete-service-$DATASET"
WORK="$RUNNER_TEMP/navhints-service-multidata-$DATASET-$GITHUB_RUN_ID"
LOCK="$ROOT/speed-device.lock"
REV="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
CAT_AUTHOR="a16eddd34b4339db5ec86e292470ce7929179bc3"
mkdir -p "$ART" "$WORK" "$ROOT"
case "$DATASET" in
  coveo)
    BASE_ROOT="$ROOT/coveo-real-demand-v1"
    INDEX_PREFIX="$BASE_ROOT/index/diskann-index"
    IVF="$BASE_ROOT/navhints-final-core-20261011/hints-b14464-nlist512.bin"
    MANIFEST="$BASE_ROOT/split/dataset.manifest.json"
    TRAIN_SOURCE="$BASE_ROOT/prepared/coveo-search-chronological-cosine.fbin"
    REPLAY_GT="$BASE_ROOT/split/replay.gt"
    ;;
  text2image-10M|bigann-10M)
    BASE_ROOT="$ROOT/public-eval/$DATASET"
    INDEX_PREFIX="$BASE_ROOT/index/diskann-index"
    IVF="$BASE_ROOT/navhints/hints-b14464-nlist512.bin"
    MANIFEST="$BASE_ROOT/data/dataset.manifest.json"
    ;;
  *) echo "unsupported frozen workload: $DATASET" >&2; exit 2 ;;
esac

for file in "$MANIFEST" "$IVF" "$IVF.manifest.json" "$INDEX_PREFIX.complete.json"; do
  test -s "$file" || { echo "missing frozen input $file" >&2; exit 3; }
done
test -d "$(dirname "$INDEX_PREFIX")"

exec 9>"$LOCK"
echo "Waiting for exclusive SSD+CPU benchmark lock $(date -Is)" | tee "$ART/lock.log"
flock 9
echo "Acquired $(date -Is)" | tee -a "$ART/lock.log"
git rev-parse HEAD > "$ART/geoivf-source-sha.txt"
printf "%s\n" "$REV" > "$ART/diskann-source-sha.txt"
printf "%s\n" "$CAT_AUTHOR" > "$ART/catapult-source-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; lsblk; } > "$ART/host.txt"
sha256sum "$MANIFEST" "$IVF" "$IVF.manifest.json" "$INDEX_PREFIX.complete.json" > "$ART/source-data-sha256.txt"
cp "$MANIFEST" "$ART/dataset.manifest.json"
cp "$IVF.manifest.json" "$ART/nav-directory.manifest.json"

if [ "$DATASET" = "coveo" ]; then
  for file in "$TRAIN_SOURCE" "$REPLAY_GT"; do test -s "$file"; done
  TRAIN="$WORK/train5000.fbin"
  python3 - "$MANIFEST" "$TRAIN_SOURCE" "$TRAIN" <<'PY'
import json,struct,sys
from pathlib import Path
manifest,full,out=map(Path,sys.argv[1:])
m=json.loads(manifest.read_text())
assert m["split"]["static_train"]==[0,5000] and m["split"]["online_replay"]==[5000,30000]
assert int(m["split"]["online_warmup_rows"])==20000 and int(m["split"]["measured_rows"])==5000
with full.open("rb") as f:
    header=f.read(8)
    rows,d=struct.unpack("<II",header)
    assert rows>=30000 and d==50, (rows,d)
    with out.open("wb") as g:
        g.write(struct.pack("<II",5000,d))
        g.write(f.read(5000*d*4))
assert out.stat().st_size==8+5000*50*4
PY
  REPLAY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["replay"])' "$MANIFEST")"
else
  readarray -t paths < <(python3 - "$MANIFEST" "$DATASET" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m["dataset"]==sys.argv[2]
assert m["split"]["train"]==[0,5000] and m["split"]["heldout"]==[5000,10000]
print(m["files"]["train5000"])
print(m["files"]["heldout5000"])
print(m["files"]["heldout5000_gt"])
PY
  )
  TRAIN="${paths[0]}"
  REPLAY="${paths[1]}"
  REPLAY_GT="${paths[2]}"
fi
for file in "$TRAIN" "$REPLAY" "$REPLAY_GT"; do
  test -s "$file" || { echo "missing frozen file $file" >&2; exit 4; }
done
sha256sum "$TRAIN" "$REPLAY" "$REPLAY_GT" >> "$ART/source-data-sha256.txt"

# Cache toolchains, but always patch the same pinned source checkout.
export CARGO_HOME="$ROOT/catapult-fidelity-cargo"
export RUSTUP_HOME="$ROOT/catapult-fidelity-rustup"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
fi
rustup toolchain install 1.97.1 --profile minimal > "$ART/rustup.log" 2>&1

for engine in nav catapult; do
  git clone --filter=blob:none https://github.com/microsoft/DiskANN.git "$WORK/$engine"
  git -C "$WORK/$engine" checkout --detach "$REV"
  test "$(git -C "$WORK/$engine" rev-parse HEAD)" = "$REV"
  if [ "$engine" = "nav" ]; then
    python3 scripts/patch_diskann_hint_ivf_packed_direct.py "$WORK/$engine"
    python3 scripts/patch_diskann_experience_core.py "$WORK/$engine"
    python3 scripts/patch_navhints_regional_history.py "$WORK/$engine"
  else
    python3 scripts/patch_diskann_catapult_snapshot.py "$WORK/$engine"
    python3 scripts/align_catapult_author_semantics.py "$WORK/$engine"
    python3 scripts/patch_catapult_author_causal_eval.py "$WORK/$engine"
  fi
  python3 scripts/patch_diskann_full_service.py "$WORK/$engine" --engine "$engine"
  git -C "$WORK/$engine" diff --check
  git -C "$WORK/$engine" diff > "$ART/$engine-service.patch"
  (
    cd "$WORK/$engine"
    cargo +1.97.1 build --release --locked -p diskann-benchmark --features disk-index \
      > "$ART/$engine-build.log" 2>&1
  )
  test -x "$WORK/$engine/target/release/diskann-benchmark"
done
NAV_BIN="$WORK/nav/target/release/diskann-benchmark"
CAT_BIN="$WORK/catapult/target/release/diskann-benchmark"
sha256sum "$NAV_BIN" "$CAT_BIN" > "$ART/native-binaries-sha256.txt"

python3 scripts/qualify_navhints_service_multidata.py \
  --dataset "$DATASET" --nav-bin "$NAV_BIN" --cat-bin "$CAT_BIN" \
  --train-queries "$TRAIN" --replay "$REPLAY" --replay-gt "$REPLAY_GT" \
  --index-prefix "$INDEX_PREFIX" --ivf "$IVF" \
  --work "$WORK/eval" --out "$ART/results" \
  --widths "${SERVICE_WIDTHS:-40,160}" \
  --reps "${SERVICE_REPS:-1}" --cat-seeds "${SERVICE_CAT_SEEDS:-0}" \
  > "$ART/evaluation.log" 2>&1

test -s "$ART/results/complete-service-$DATASET.json"
grep -F "COMPLETE_SERVICE_MULTIDATA_PASS dataset=$DATASET" "$ART/evaluation.log"
echo "COMPLETE_SERVICE_MULTIDATA_PASS:$DATASET" | tee "$ART/status.txt"
