#!/usr/bin/env bash
set -euxo pipefail

TARGET_BYTES=3159860
STARLING_REV=17dc3e8a011533a62374445f53963e951b72883a
DISKANN_REV=fcf90534174cf29c78c9f13b4cccf1fcabff85f5

mkdir -p artifacts/memgraph-baseline
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy==2.2.6

BASE="$HOME/.cache/geoivf/catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
QUERIES="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
GT="$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
INDEX_PREFIX="$HOME/.cache/geoivf/catapult-paper-pubmed1m-index-v1/diskann-index"
ROUTER="$HOME/.cache/geoivf/region-ablation-routers-v1/nlist-512/router.bin"
WAYPOINT="$HOME/.cache/geoivf/region-ablation-caches-v1/nlist-512/heavy-skip-b2500.bin"
TRACE_CACHE="$HOME/.cache/geoivf/memgraph-medoid-trace-v1"
MEM_CACHE="$HOME/.cache/geoivf/starling-memgraph-3mib-v1"

for f in "$BASE" "$QUERIES" "$GT" "$ROUTER" "$WAYPOINT"; do test -s "$f"; done
mkdir -p "$TRACE_CACHE" "$MEM_CACHE"
git rev-parse HEAD > artifacts/memgraph-baseline/geoivf-commit.txt
{ uname -a; lscpu; free -h; df -h; } > artifacts/memgraph-baseline/host.txt

.venv/bin/python -m py_compile   scripts/collect_medoid_navigation_traces.py   scripts/prepare_starling_memgraph_sample.py   scripts/summarize_starling_memgraph.py   scripts/qualify_memgraph_baseline.py

export CARGO_HOME="$RUNNER_TEMP/memgraph-cargo"
export RUSTUP_HOME="$RUNNER_TEMP/memgraph-rustup"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
curl -fsSL --retry 3 https://sh.rustup.rs -o "$RUNNER_TEMP/memgraph-rustup.sh"
sh "$RUNNER_TEMP/memgraph-rustup.sh" -y --profile minimal --no-modify-path --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt

rm -rf third_party/DiskANN-memgraph-trace
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-memgraph-trace
git -C third_party/DiskANN-memgraph-trace checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_waypoint_trace.py third_party/DiskANN-memgraph-trace
(cd third_party/DiskANN-memgraph-trace && cargo fmt --all)
git -C third_party/DiskANN-memgraph-trace diff --check
(cd third_party/DiskANN-memgraph-trace && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/memgraph-baseline/trace-build.log
TRACE_BIN="$PWD/third_party/DiskANN-memgraph-trace/target/release/diskann-benchmark"

TRAIN_TRACE="$TRACE_CACHE/medoid-k1-train5000-trace.jsonl"
if [ ! -s "$TRAIN_TRACE" ]; then
  rm -rf "$TRACE_CACHE"
  mkdir -p "$TRACE_CACHE"
  TRACE_WORK="$RUNNER_TEMP/memgraph-trace-work"
  rm -rf "$TRACE_WORK"; mkdir -p "$TRACE_WORK"
  .venv/bin/python scripts/collect_medoid_navigation_traces.py     --binary "$TRACE_BIN" --queries "$QUERIES" --gt "$GT"     --index-prefix "$INDEX_PREFIX" --work "$TRACE_WORK"     --out "$TRACE_CACHE" --train-rows 5000     2>&1 | tee artifacts/memgraph-baseline/trace.log
  rm -rf "$TRACE_WORK"
fi
test -s "$TRAIN_TRACE"
cp "$TRACE_CACHE/trace-summary.json" artifacts/memgraph-baseline/trace-summary.json

rm -rf third_party/starling
git clone --recursive https://github.com/zilliztech/starling.git third_party/starling
git -C third_party/starling checkout --detach "$STARLING_REV"
git -C third_party/starling submodule update --init --recursive
echo "$STARLING_REV" > artifacts/memgraph-baseline/starling-commit.txt
cmake -S third_party/starling -B third_party/starling/build -DCMAKE_BUILD_TYPE=Release   2>&1 | tee artifacts/memgraph-baseline/starling-cmake.log
cmake --build third_party/starling/build --target build_memory_index search_memory_index -j4   2>&1 | tee artifacts/memgraph-baseline/starling-build.log
STARLING_BUILD="$PWD/third_party/starling/build"

HELDOUT="$RUNNER_TEMP/memgraph-heldout.fbin"
.venv/bin/python - "$QUERIES" "$HELDOUT" <<'PY'
import struct,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
with src.open('rb') as f:
    rows,dim=struct.unpack('<II',f.read(8))
    assert (rows,dim)==(10000,768)
    f.seek(8+5000*dim*4)
    body=f.read(5000*dim*4)
    assert len(body)==5000*dim*4
with dst.open('wb') as f:
    f.write(struct.pack('<II',5000,dim))
    f.write(body)
PY

BUILDER="$STARLING_BUILD/tests/build_memory_index"
SEARCHER="$STARLING_BUILD/tests/search_memory_index"
test -x "$BUILDER"; test -x "$SEARCHER"

for mode in frequency uniform; do
  ROOT="$MEM_CACHE/$mode"
  mkdir -p "$ROOT"
  CHOSEN="$ROOT/CHOSEN-v1.json"
  rm -f "$CHOSEN"
  found=0

  for count in 1024 1000 980 960 940 920 900 880 860 840 820 800; do
    SAMPLE="$ROOT/sample-$count"
    IDX="$ROOT/index-$count"
    rm -f "$SAMPLE"_data.bin "$SAMPLE"_ids.bin "$SAMPLE".manifest.json "$IDX"*

    if [ "$mode" = frequency ]; then
      .venv/bin/python scripts/prepare_starling_memgraph_sample.py         --base "$BASE" --trace "$TRAIN_TRACE" --out-prefix "$SAMPLE"         --count "$count" --mode frequency         > "artifacts/memgraph-baseline/sample-$mode-$count.log" 2>&1
    else
      .venv/bin/python scripts/prepare_starling_memgraph_sample.py         --base "$BASE" --out-prefix "$SAMPLE" --count "$count"         --mode uniform --seed 12345         > "artifacts/memgraph-baseline/sample-$mode-$count.log" 2>&1
    fi

    "$BUILDER" --data_type float --dist_fn mips --data_path "$SAMPLE"       --index_path_prefix "$IDX" -R 48 -L 128 --alpha 1.2 -T 4       > "artifacts/memgraph-baseline/build-$mode-$count.log" 2>&1

    bytes=$(.venv/bin/python - "$IDX" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
files=[x for x in p.parent.glob(p.name+'*') if x.is_file()]
if not files: raise SystemExit('no Starling index files')
print(sum(x.stat().st_size for x in files))
PY
)
    echo "$mode count=$count bytes=$bytes" | tee -a artifacts/memgraph-baseline/memory-sweep.log
    if [ "$bytes" -le "$TARGET_BYTES" ]; then
      .venv/bin/python - "$CHOSEN" "$count" "$IDX" "$bytes" "$TARGET_BYTES" <<'PY'
import json,sys
path,count,idx,bs,target=sys.argv[1:]
json.dump({"count":int(count),"index_prefix":idx,"index_bytes":int(bs),"target_bytes":int(target)},open(path,'w'),indent=2)
PY
      found=1
      break
    fi
    rm -f "$IDX"*
  done
  test "$found" = 1

  count=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["count"])' "$CHOSEN")
  idx=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_prefix"])' "$CHOSEN")
  bytes=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_bytes"])' "$CHOSEN")
  test "$bytes" -le "$TARGET_BYTES"

  RESULT="$PWD/artifacts/memgraph-baseline/starling-$mode"
  LOG="$PWD/artifacts/memgraph-baseline/starling-$mode-search.log"
  "$SEARCHER" --data_type float --dist_fn mips --index_path_prefix "$idx"     --query_file "$HELDOUT" --gt_file null -K 1     -L 1 4 8 16 32 64 --result_path "$RESULT" -T 4 --tags true     2>&1 | tee "$LOG"

  .venv/bin/python scripts/summarize_starling_memgraph.py     --mode "$mode" --sample-count "$count" --index-prefix "$idx"     --search-log "$LOG" --result-prefix "$RESULT"     --search-ls "1,4,8,16,32,64" --target-bytes "$TARGET_BYTES"     --out "$PWD/artifacts/memgraph-baseline/starling-$mode-manifest.json"

  cp "$CHOSEN" "artifacts/memgraph-baseline/chosen-$mode.json"
done

rm -rf third_party/DiskANN-memgraph-start
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-memgraph-start
git -C third_party/DiskANN-memgraph-start checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_start_points.py third_party/DiskANN-memgraph-start
(cd third_party/DiskANN-memgraph-start && cargo fmt --all)
git -C third_party/DiskANN-memgraph-start diff --check
(cd third_party/DiskANN-memgraph-start && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/memgraph-baseline/start-build.log
START_BIN="$PWD/third_party/DiskANN-memgraph-start/target/release/diskann-benchmark"

rm -rf third_party/DiskANN-memgraph-waypoint
git clone --filter=blob:none https://github.com/microsoft/DiskANN.git third_party/DiskANN-memgraph-waypoint
git -C third_party/DiskANN-memgraph-waypoint checkout --detach "$DISKANN_REV"
.venv/bin/python scripts/patch_diskann_waypoint_cache.py third_party/DiskANN-memgraph-waypoint
(cd third_party/DiskANN-memgraph-waypoint && cargo fmt --all)
git -C third_party/DiskANN-memgraph-waypoint diff --check
(cd third_party/DiskANN-memgraph-waypoint && cargo build --release --locked -p diskann-benchmark --features disk-index)   2>&1 | tee artifacts/memgraph-baseline/waypoint-build.log
WAYPOINT_BIN="$PWD/third_party/DiskANN-memgraph-waypoint/target/release/diskann-benchmark"

COMPARE_WORK="$RUNNER_TEMP/memgraph-compare"
rm -rf "$COMPARE_WORK"; mkdir -p "$COMPARE_WORK" artifacts/memgraph-baseline/results
.venv/bin/python scripts/qualify_memgraph_baseline.py   --start-binary "$START_BIN" --waypoint-binary "$WAYPOINT_BIN"   --queries "$QUERIES" --gt "$GT" --index-prefix "$INDEX_PREFIX"   --router "$ROUTER" --waypoint "$WAYPOINT"   --memgraph-manifest "$PWD/artifacts/memgraph-baseline/starling-frequency-manifest.json"   --memgraph-manifest "$PWD/artifacts/memgraph-baseline/starling-uniform-manifest.json"   --work "$COMPARE_WORK" --out "$PWD/artifacts/memgraph-baseline/results"   --train-rows 5000   2>&1 | tee artifacts/memgraph-baseline/eval.log

rm -rf "$COMPARE_WORK" "$HELDOUT"
