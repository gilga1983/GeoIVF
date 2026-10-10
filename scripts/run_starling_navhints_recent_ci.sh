#!/usr/bin/env bash
# BigANN-10M native author-system comparison: real original 10M base, same
# frozen 5K heldout set (first 4K storage warm, final 1K measured) as NavHints.
# These index layouts are author-native rather than artificially transplanted
# onto Microsoft DiskANN. Do not label cross-backend measurements paired.
set -euo pipefail
set -x

ART="$PWD/artifacts/starling-navhints-recent"
WORK="$RUNNER_TEMP/starling-navhints-recent-$GITHUB_RUN_ID"
DATA="$WORK/data"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
SRC="$HOME/.cache/geoivf/public-eval/bigann-10M/data/dataset.manifest.json"
mkdir -p "$ART" "$WORK" "$DATA" "$(dirname "$LOCK")"
git rev-parse HEAD > "$ART/geoivf-revision.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
test -s "$SRC"

# Use the existing pinned public BigANN-10M base and official ground truth.
# No downsampling and no ground-truth recomputation.
python3 - "$SRC" "$DATA" <<'PY'
import json, struct, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
assert m["dataset"]=="bigann-10M" and m["data_type"]=="uint8"
assert m["dim"]==128 and m["metric"]=="squared_l2"
srcbase=Path(m["files"]["base"]).resolve()
srcq=Path(m["files"]["heldout5000"]).resolve()
srcgt=Path(m["files"]["heldout5000_gt"]).resolve()
def sh(p):
    with p.open("rb") as f: return struct.unpack("<II",f.read(8))
assert sh(srcbase)==(10_000_000,128)
assert sh(srcq)==(5_000,128)
assert sh(srcgt)==(5_000,100)
assert srcbase.stat().st_size==8+10_000_000*128
base=out/"base.u8bin"; base.symlink_to(srcbase)
def query_slice(dst,start,count):
    with srcq.open("rb") as fi, dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,128))
        fi.seek(8+start*128)
        b=fi.read(count*128)
        assert len(b)==count*128
        fo.write(b)
def gt_slice(dst,start,count):
    k=100; rows=5000; stride=100*4
    with srcgt.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        for off in (8+start*stride,8+rows*stride+start*stride):
            fi.seek(off)
            b=fi.read(count*stride)
            assert len(b)==count*stride
            fo.write(b)
query_slice(out/"warm4000.u8bin",0,4000)
query_slice(out/"eval1000.u8bin",4000,1000)
gt_slice(out/"warm4000.gt",0,4000)
gt_slice(out/"eval1000.gt",4000,1000)
summary={
  "name":"official-author-systems BigANN-10M comparison",
  "base_vectors":10_000_000, "dim":128,"metric":"squared_l2",
  "index_builds":"independent native graph layouts, not same-graph controls",
  "query_contract":"heldout5000 rows 0:4000 warm then 4000:5000 timed",
  "search_threads":4,"beam_width":8,
  "native_navigation_memory":"separately recorded; larger than NavHints",
  "source_manifest":str(Path(sys.argv[1]).resolve()),
  "reference_checksums":m.get("sha256",{}),
  "files":{"base":str(base),"warm":str(out/"warm4000.u8bin"),
           "eval":str(out/"eval1000.u8bin")}
}
(out/"bigann10m.manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
PY
cp "$DATA/bigann10m.manifest.json" "$ART/"
# Preserve *inner* author-implementation build and layout logs on any failure.
# The launcher itself redirects its graph-partition logs into the native
# index directory, so without this hook failures are otherwise opaque.
capture_failure_diagnostics() {
  local code=$?
  if [[ "$code" -eq 0 ]]; then return; fi
  set +e
  mkdir -p "$ART/phase-logs"
  python3 - "$DATA" "$ART/phase-logs" <<'PY_DIAG'
from pathlib import Path
import json
import shutil
import struct
import sys

root, dest = map(Path, sys.argv[1:])
summary=[]
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.name in {"relayout.log", "_part.bin.log", "build.log"}:
        relative=path.relative_to(root)
        dst=dest/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst)
    if path.name != "_part.bin":
        continue
    result={"path":str(path.relative_to(root)),"bytes":path.stat().st_size}
    try:
        with path.open("rb") as f:
            head=f.read(24)
            if len(head)!=24: raise ValueError("truncated partition header")
            cap, num_parts, num_nodes = struct.unpack("<QQQ",head)
            result.update(capacity=cap,partitions=num_parts,nodes=num_nodes)
            if num_parts>20_000_000: raise ValueError("implausibly many partitions")
            empty=bad_first=overflow=invalid_ids=0
            count=0
            for i in range(num_parts):
                n_bytes=f.read(4)
                if len(n_bytes)!=4: raise ValueError(f"truncated block {i}")
                (size,)=struct.unpack("<I",n_bytes)
                if size>100_000: raise ValueError(f"implausible block size at {i}: {size}")
                blob=f.read(size*4)
                if len(blob)!=size*4: raise ValueError(f"truncated layout at {i}")
                ids=struct.unpack(f"<{size}I",blob)
                empty+=not size
                bad_first+=bool(size and ids[0]!=i)
                overflow+=bool(size>cap)
                invalid_ids+=sum(x>=num_nodes for x in ids)
                count+=1
            result.update(parsed=count,empty_partitions=empty,
                          first_id_mismatches=bad_first,oversized_partitions=overflow,
                          invalid_vertex_ids=invalid_ids)
    except Exception as exc:
        result["error"]=repr(exc)
    summary.append(result)
(dest/"partition-check.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
PY_DIAG

  # Read-only debugger replay of the exact official relayout utility.
  # Optional: no system modification or C++ source patch.
  if command -v gdb >/dev/null 2>&1; then
    local bin="$DATA/gorgeous/release/tests/utils/index_relayout_free_mem"
    local gp
    gp="$(find "$DATA/gorgeous" -name '_part.bin' -path '*/GRAPH_CACHE_INDEX/*' -print -quit 2>/dev/null)"
    local idx
    idx="$(find "$DATA/gorgeous" -name '_disk_beam_search.index' -print -quit 2>/dev/null)"
    if [[ -x "$bin" && -s "$gp" && -s "$idx" ]]; then
      gdb -q -batch -ex 'set pagination off' -ex run -ex bt \
        --args "$bin" "$idx" "$gp" uint8 3 4096 4096 \
        >"$ART/phase-logs/gorgeous-relayout-backtrace.txt" 2>&1 || true
    fi
  else
    printf '%s\n' 'gdb absent on self-hosted runner; partition and relayout logs saved' \
      >"$ART/phase-logs/gdb-status.txt"
  fi
}
trap capture_failure_diagnostics EXIT

# Serializing native C++ compilation as well as I/O avoids disturbing the
# already-running real-NVMe memory sweeps on this shared self-hosted machine.
exec 9>"$LOCK"
echo "Waiting for exclusive shared-device/CPU lock" | tee "$ART/lock.log"
flock 9
echo "Acquired lock $(date -Is)" | tee -a "$ART/lock.log"

build_system() {
    local sys="$1" url="$2" rev="$3"
    local root="$WORK/$sys"
    echo "Starting official $sys at $rev" | tee "$ART/$sys-source.txt"
    git clone --filter=blob:none "$url" "$root"
    git -C "$root" checkout --detach "$rev"
    git -C "$root" submodule update --init --recursive --jobs 2
    test "$(git -C "$root" rev-parse HEAD)" = "$rev"
    git -C "$root" submodule status > "$ART/$sys-submodules.txt"

    # Only dataset config and author-provided launcher change. Never patch
    # graph/search C++ code, and never require sudo to drop the OS page cache.
    cat > "$root/scripts/config_dataset.sh" <<EOF
DATA_DIR="$DATA"
dataset_pilot() {
  BASE_PATH="$DATA/base.u8bin"
  QUERY_FILE="$DATA/eval1000.u8bin"
  GT_FILE="$DATA/eval1000.gt"
  PREFIX="bigann_native_10m"
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=10000000
  SECTOR_LEN=4096
  GR_SECTOR_LEN=4096
  N_PQ_CODE=4
}
EOF

    if [ "$sys" = "starling" ]; then
        cat > "$root/scripts/config_local.sh" <<'EOF'
source config_dataset.sh
dataset_pilot
R=48
BUILD_L=128
M=24
BUILD_T=4
USE_SQ=0
MEM_R=24
MEM_BUILD_L=100
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
MEM_USE_FREQ=0
MEM_FREQ_USE_RATE=0.01
GP_TIMES=16
GP_T=4
GP_LOCK_NUMS=0
GP_USE_FREQ=0
GP_CUT=4096
BM_LIST=(8)
T_LIST=(4)
CACHE=0
MEM_L=10
MEM_TOPK=10
USE_PAGE_SEARCH=1
PS_USE_RATIO=1.0
LS="20 40 80 160 320 640"
EOF
    else
        cat > "$root/scripts/config_local.sh" <<'EOF'
source config_dataset.sh
dataset_pilot
R=64
BUILD_L=128
M=24
BUILD_T=4
MEM_R=24
MEM_BUILD_L=100
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
GP_TIMES=16
GP_T=4
GP_LOCK_NUMS=0
GP_CUT=4096
BM_LIST=(8)
T_LIST=(4)
CACHE=0
MEM_L=10
DECO_IMPL=1
MEM_GRAPH_USE_RATIO=0.2
MEM_EMB_USE_RATIO=0.0
EMB_SEARCH_RATIO=0.4
USE_DISK_GRAPH_CACHE_INDEX=1
PQ_FILTER_RATIO=0.9
USE_PAGE_SEARCH=1
PS_USE_RATIO=0.3
LS="20 40 80 160 320 640"
EOF
    fi

    python3 - "$root/scripts/run_benchmark.sh" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
s=p.read_text()
# CMake 4+ no longer accepts oneTBB's legacy policy minimum.
# Apply compatibility only to this temporary author launcher, not C++ code.
cmake_marker="cmake -DCMAKE_BUILD_TYPE="
assert cmake_marker in s, "Author CMake invocation changed unexpectedly"
s=s.replace(cmake_marker, "cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=")
# Resource limit: no unbounded make -j on a shared benchmark host.
s=s.replace("make -j\n","make -j4\n")
# Starling's launcher drops kernel page cache via sudo. The experiment
# preserves cache warmth for every method and does not use elevated access.
s=s.replace("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;","sync;")
p.write_text(s)
PY
    if [ "$sys" = "starling" ]; then
        python3 scripts/patch_starling_recent_hints.py "$root"
        git -C "$root" diff --check
        git -C "$root" diff -- include/pq_flash_index.h src/page_search.cpp tests/search_disk_index.cpp > "$ART/$sys-navhints-code.diff"
    fi
    git -C "$root" diff -- scripts/config_dataset.sh scripts/config_local.sh scripts/run_benchmark.sh > "$ART/$sys-launcher.diff" || true

    if [ "$sys" = "gorgeous" ]; then
        # Correct an upstream sector-count bug in the *layout writer*, not
        # its partitioning/search algorithms. The destination partition size
        # C differs from the original index's nnodes_per_sector. Using C
        # under-reads the index and dereferences beyond mem_index in memcpy.
        python3 - "$root/tests/utils/index_relayout_free_mem.cpp" <<'PY_FIX'
import sys
from pathlib import Path
p=Path(sys.argv[1])
s=p.read_text()
old="auto diskann_partition_number = ROUND_UP(_nd, C) / C;"
new="auto diskann_partition_number = ROUND_UP(_nd, nnodes_per_sector) / nnodes_per_sector;"
assert s.count(old)==1, "unexpected Gorgeous layout source; refuse non-exact patch"
p.write_text(s.replace(old,new))
print("Fixed original-index input sector count, no navigation/layout policy change", flush=True)
PY_FIX
        git -C "$root" diff --check
        git -C "$root" diff -- tests/utils/index_relayout_free_mem.cpp > "$ART/$sys-relayout-sector-count-fix.diff"
    fi

    ( cd "$root/scripts"
      bash run_benchmark.sh release build
      bash run_benchmark.sh release build_mem
      if [ "$sys" = "starling" ]; then
          bash run_benchmark.sh release gp
      else
          bash run_benchmark.sh release split_graph
          bash run_benchmark.sh release gr_layout
      fi
    ) > "$ART/$sys-build-layout.log" 2>&1
    echo "Completed author-native $sys index and disk layout $(date -Is)" | tee "$ART/$sys-build-success.txt"
    du -sh "$root/indices" "$DATA/gorgeous" 2>/dev/null | tee "$ART/$sys-build-disk-usage.txt" || true
}
build_system "starling" "https://github.com/zilliztech/starling.git" "17dc3e8a011533a62374445f53963e951b72883a"

# Test queries are original 5K held-out (first 4K causal warm-up).
python3 - "$SRC" "$DATA" <<'PY_HELDOUT'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2]);files=m["files"]
for key,dest in [("heldout5000","heldout5000.u8bin"),("heldout5000_gt","heldout5000.gt")]:
    target=Path(files[key]).resolve()
    target_path=out/dest
    if target_path.exists() or target_path.is_symlink():target_path.unlink()
    target_path.symlink_to(target)
    assert target_path.is_file()
print("STARLING_HELDOUT_5K_VERIFIED")
PY_HELDOUT

source_dir="$WORK/starling"
config="$source_dir/scripts/config_dataset.sh"
sed -i "s#QUERY_FILE=.*#QUERY_FILE=$DATA/heldout5000.u8bin#" "$config"
sed -i "s#GT_FILE=.*#GT_FILE=$DATA/heldout5000.gt#" "$config"
grep -E '(QUERY_FILE|GT_FILE)=' "$config" >"$ART/starling-query-config.txt"
test -s "$ART/starling-navhints-code.diff"

# Same native navigator and sequential 4K-warm/1K-eval query loop, each arm.
for rep in 0 1 2; do
    case "$rep" in
      0) order=(baseline random512 recent512) ;;
      1) order=(recent512 baseline random512) ;;
      2) order=(random512 recent512 baseline) ;;
    esac
    for mode in "${order[@]}"; do
        echo "STARLING_HINT_EXPERIMENT rep=$rep mode=$mode begin=$(date -Is)"
        (cd "$source_dir/scripts" && STARLING_NAVHINTS_EVAL="$mode" bash run_benchmark.sh release search knn) \
            >"$ART/starling-rep$rep-$mode-execution.log" 2>&1
        grep -E 'Recall@10|^[[:space:]]*(20|40|80|160|320|640)[[:space:]]' \
            "$ART/starling-rep$rep-$mode-execution.log" | tail -n 15 \
            >"$ART/starling-rep$rep-$mode-summary.txt"
        test "$(grep -E -c '^[[:space:]]*(20|40|80|160|320|640)[[:space:]]' \
            "$ART/starling-rep$rep-$mode-summary.txt")" -ge 6
        # The author launcher writes native binary output to its own log.
        # Validate that exact mode and the full 5K causal replay there.
        search_log=$(grep 'Searching... log file:' \
            "$ART/starling-rep$rep-$mode-execution.log" | tail -n 1 | sed -E 's/^.*log file: //')
        # Native Starling log paths are relative to the author scripts directory.
        # Copy the original detailed search/recall results into the artifact.
        if [ ! -s "$source_dir/indices/$search_log" ]; then
          echo "STARLING_NATIVE_LOG_MISSING mode=$mode path=$source_dir/scripts/$search_log" >&2
          exit 8
        fi
        cp "$source_dir/indices/$search_log" "$ART/starling-rep$rep-$mode-native-search.log"
        search_log="$ART/starling-rep$rep-$mode-native-search.log"
        grep -q "STARLING_NAVHINTS_PROTOCOL mode=$mode warmup=4000 measured=1000" "$search_log"
        cat "$ART/starling-rep$rep-$mode-summary.txt"
        echo "STARLING_HINT_EXPERIMENT rep=$rep mode=$mode complete=$(date -Is)"
    done
done
printf '%s\n' "STARLING_NAVHINTS_RECENT512_PILOT_SUCCESS" | tee "$ART/result.txt"
