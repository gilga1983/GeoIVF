#!/usr/bin/env bash
# Author-source integration pilot, NOT a publishable performance comparison.
# Build complete Starling and Gorgeous configurations on the same tiny BigANN
# subset and verify native Recall@10. Official 10M evaluation follows only
# after this passes and an independently audited resource/recall contract exists.
set -euo pipefail
set -x

ART="$PWD/artifacts/native-ann-pilot"
WORK="$RUNNER_TEMP/native-ann-pilot-$GITHUB_RUN_ID"
DATA="$WORK/data"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
SRC="$HOME/.cache/geoivf/public-eval/bigann-10M/data/dataset.manifest.json"
mkdir -p "$ART" "$WORK" "$DATA" "$(dirname "$LOCK")"
git rev-parse HEAD > "$ART/geoivf-revision.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
test -s "$SRC"

# Keep dependency state and all generated graphs out of source checkout/cache.
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check 'numpy==2.2.6' 'faiss-cpu>=1.9,<2'

"$WORK/venv/bin/python" - "$SRC" "$DATA" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path
import faiss
import numpy as np

manifest=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2])
assert manifest["dataset"] == "bigann-10M"
assert manifest["data_type"] == "uint8"
assert manifest["dim"] == 128
assert manifest["metric"] == "squared_l2"

def shape(path):
    with open(path,"rb") as f:
        return struct.unpack("<II",f.read(8))
base=Path(manifest["files"]["base"])
held=Path(manifest["files"]["heldout5000"])
assert shape(base)==(10_000_000,128)
assert shape(held)==(5_000,128)
assert base.stat().st_size==8+10_000_000*128
rows=20_000
questions=200
b=np.memmap(base,dtype=np.uint8,mode="r",offset=8,shape=(10_000_000,128))
q=np.memmap(held,dtype=np.uint8,mode="r",offset=8,shape=(5_000,128))
vectors=np.array(b[:rows])
queries=np.array(q[:questions])
fout=out/"base.u8bin"
with fout.open("wb") as f:
    f.write(struct.pack("<II",rows,128))
    f.write(vectors.tobytes())
qout=out/"eval.u8bin"
with qout.open("wb") as f:
    f.write(struct.pack("<II",questions,128))
    f.write(queries.tobytes())
idx=faiss.IndexFlatL2(128)
idx.add(vectors.astype(np.float32))
dists,ids=idx.search(queries.astype(np.float32),10)
truth=out/"eval.gt"
with truth.open("wb") as f:
    f.write(struct.pack("<II",questions,10))
    f.write(ids.astype("<u4").tobytes())
    f.write(dists.astype("<f4").tobytes())
summary={
    "purpose":"Native author-code CLI/index-build pilot only; never compare with 10M paper results",
    "dataset":"bigann-10M FIRST 20000 vectors, 200 heldout queries",
    "metric":"squared_l2","data_type":"uint8","dim":128,
    "original_dataset_manifest":str(Path(sys.argv[1]).resolve()),
    "data":{"base":str(fout),"queries":str(qout),"gt":str(truth)},
    "sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (fout,qout,truth)},
}
(out/"pilot.manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
PY
cp "$DATA/pilot.manifest.json" "$ART/"

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
  QUERY_FILE="$DATA/eval.u8bin"
  GT_FILE="$DATA/eval.gt"
  PREFIX="bigann_native_pilot"
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.0006
  K=10
  DATA_DIM=128
  DATA_N=20000
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
M=4
BUILD_T=4
USE_SQ=0
MEM_R=24
MEM_BUILD_L=100
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
MEM_USE_FREQ=0
MEM_FREQ_USE_RATE=0.01
GP_TIMES=2
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
LS="40 80 160"
EOF
    else
        cat > "$root/scripts/config_local.sh" <<'EOF'
source config_dataset.sh
dataset_pilot
R=64
BUILD_L=128
M=4
BUILD_T=4
MEM_R=24
MEM_BUILD_L=100
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.01
GP_TIMES=2
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
LS="40 80 160"
EOF
    fi

    python3 - "$root/scripts/run_benchmark.sh" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
s=p.read_text()
# Resource limit: no unbounded make -j on a shared benchmark host.
assert "make -j" in s
s=s.replace("make -j\n","make -j4\n")
# Starling's launcher drops kernel page cache via sudo. The experiment
# preserves cache warmth for every method and does not use elevated access.
s=s.replace("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;","sync;")
p.write_text(s)
PY
    git -C "$root" diff -- scripts/config_dataset.sh scripts/config_local.sh scripts/run_benchmark.sh > "$ART/$sys-launcher.diff" || true
    ( cd "$root/scripts"
      bash run_benchmark.sh release build
      bash run_benchmark.sh release build_mem
      if [ "$sys" = "starling" ]; then
          bash run_benchmark.sh release gp
      else
          bash run_benchmark.sh release split_graph
          bash run_benchmark.sh release gr_layout
      fi
      bash run_benchmark.sh release search knn
    ) > "$ART/$sys-execution.log" 2>&1
    echo "Completed native $sys $(date -Is)" | tee "$ART/$sys-success.txt"
    grep -E 'Recall@10|^[[:space:]]*(40|80|160)[[:space:]]' "$ART/$sys-execution.log" | tail -n 14 | tee "$ART/$sys-recall-table.txt"
    test "$(grep -E -c '^[[:space:]]*(40|80|160)[[:space:]]' "$ART/$sys-execution.log")" -ge 3
}
build_system "starling" "https://github.com/zilliztech/starling.git" "17dc3e8a011533a62374445f53963e951b72883a"
build_system "gorgeous" "https://github.com/yinpeiqi/Gorgeous.git" "04c8c27d77a19e7751748d4e0ac4ecd2f0fb3e8b"
printf '%s\n' "NATIVE_COMPLETE_PILOT_SUCCESS" | tee "$ART/result.txt"
