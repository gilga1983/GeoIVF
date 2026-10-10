#!/usr/bin/env bash
# Reproduce ONLY the actual author's Rust code; not a Catapult-style patch.
# Smoke verifies source, native build, mode comparison and output schema.
set -euo pipefail
set -x
ART="$PWD/artifacts/catapult-author-smoke"
WORK="$RUNNER_TEMP/catapult-author-smoke-$GITHUB_RUN_ID"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
AUTHOR="a16eddd34b4339db5ec86e292470ce7929179bc3"
TOOLCHAIN="nightly-2026-09-15"
mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")"
git rev-parse HEAD > "$ART/geoivf-revision.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
exec 9>"$LOCK"
echo "Waiting to avoid interference with the BigANN SSD experiment" | tee "$ART/lock.txt"
flock 9
echo "Acquired exclusive machine lock $(date -Is)" | tee -a "$ART/lock.txt"
git clone --filter=blob:none https://github.com/MRandl/catapult-db.git "$WORK/catapult-db"
git -C "$WORK/catapult-db" checkout --detach "$AUTHOR"
test "$(git -C "$WORK/catapult-db" rev-parse HEAD)" = "$AUTHOR"
printf '%s\n' "$AUTHOR" > "$ART/author-original-sha.txt"
rustup toolchain install "$TOOLCHAIN" --profile minimal
(
  cd "$WORK/catapult-db"
  cargo "+$TOOLCHAIN" build --release --locked --bin run_queries
  test -s test/index/ann
  test -s test/index/ann_vectors.bin
  test -s test/index/vectors.npy
  for mode in vanilla catapult; do
    target/release/run_queries \
      --queries test/index/vectors.npy \
      --graph test/index/ann \
      --payload test/index/ann_vectors.bin \
      --mode "$mode" \
      --threads 1 --beam-width 2 --seeds 42 \
      --output "$ART/$mode.json" --output-neighbors \
      > "$ART/$mode.log" 2>&1
  done
)
python3 - "$ART" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for mode in ('vanilla','catapult'):
    obj=json.loads((root/f'{mode}.json').read_text())
    rows=obj.get('results')
    assert isinstance(rows,list) and len(rows)==1,(mode,obj)
    row=rows[0]
    assert row['num_queries']>0 and row['num_threads']==1
    assert row['catapults_enabled']==(mode=='catapult')
    assert row['qps']>0 and row['avg_dists_computed']>0
    assert len(row['neighbors'])==row['num_queries']
    print("CATAPULT_ORIGINAL_AUTHOR_SMOKE_VERIFIED", mode, "queries",row['num_queries'],"QPS",row['qps'])
print("CATAPULT_AUTHOR_PINNED_NATIVE_SMOKE_SUCCESS")
PY
