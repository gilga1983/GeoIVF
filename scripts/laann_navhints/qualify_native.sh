#!/usr/bin/env bash
# Author-LAANN + NavHints proof on real NVMe. Self-hosted only, 2 build threads.
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
ART="${GITHUB_WORKSPACE:-$PWD}/artifacts/navhints-laann-native"
CACHE="$HOME/.cache/geoivf"
WORK="${RUNNER_TEMP:-/tmp}/navhints-laann-${GITHUB_RUN_ID:-manual}"
SRC="$WORK/PageANN"
RUNNER_DATA="$CACHE/datasets/sift1m"
INDEX_CACHE="$CACHE/laann-native-sift1m/index"
PIN="58fe40feb58d0ca1a168816b36c4d4b22d3eb29c"
mkdir -p "$ART" "$WORK" "$INDEX_CACHE"
printf 'LAANN_PIN=%s\nGEOIVF_SHA=%s\n' "$PIN" "$(git rev-parse HEAD)" > "$ART/provenance.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
python3 -m py_compile scripts/laann_navhints/*.py
bash -n scripts/laann_navhints/qualify_native.sh
git clone --filter=blob:none https://github.com/Dingyi-Kang/PageANN.git "$SRC" > "$ART/clone.log" 2>&1
git -C "$SRC" checkout --detach "$PIN" >> "$ART/clone.log" 2>&1
test "$(git -C "$SRC" rev-parse HEAD)" = "$PIN"
python3 scripts/laann_navhints/patch_laann.py --source-tree "$SRC" > "$ART/patch.log" 2>&1
git -C "$SRC" diff --check
git -C "$SRC" diff -- src/pq_flash_index_laann.cpp apps/search_disk_index.cpp > "$ART/laann-navhints.patch"
sha256sum "$SRC/src/nh_laann_state.h" >> "$ART/provenance.txt"
printf 'Native LAANN nav/page graph and look-ahead pipeline enabled in all arms\n' >> "$ART/provenance.txt"

# Build liburing in the private scratch directory if the self-hosted machine
# lacks development headers. Never apt-install system packages on the Legion.
# Exact upstream release tag has Git SHA 7a7b035470175b3ecf071384274d7859885ece32.
URING_PREFIX="$WORK/liburing-private-install"
if ! printf '#include <liburing.h>\n' | c++ -x c++ -fsyntax-only - >/dev/null 2>&1; then
  git clone --filter=blob:none https://github.com/axboe/liburing.git "$WORK/liburing" \
    > "$ART/liburing-clone.log" 2>&1
  git -C "$WORK/liburing" checkout --detach 7a7b035470175b3ecf071384274d7859885ece32 \
    >> "$ART/liburing-clone.log" 2>&1
  (
    cd "$WORK/liburing"
    ./configure --prefix="$URING_PREFIX" --libdir="$URING_PREFIX/lib"
    nice -n 10 make -j2
    make install
  ) > "$ART/liburing-build.log" 2>&1
  test -s "$URING_PREFIX/include/liburing.h"
  test -s "$URING_PREFIX/lib/liburing.a"
  export CPATH="$URING_PREFIX/include:${CPATH:-}"
  export LIBRARY_PATH="$URING_PREFIX/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$URING_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  printf 'PRIVATE_LIBURING_SHA=%s\n' 7a7b035470175b3ecf071384274d7859885ece32 >> "$ART/provenance.txt"
fi
# Author build scripts and exact pinned source, no installation of system packages.
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INCLUDE_PATH="$URING_PREFIX/include" \
  -DCMAKE_LIBRARY_PATH="$URING_PREFIX/lib" > "$ART/cmake.log" 2>&1
nice -n 10 cmake --build "$SRC/build" --parallel 2 > "$ART/build.log" 2>&1
BIN="$SRC/build/apps/search_disk_index"
test -x "$BIN"
"$BIN" --help > "$ART/native-help.txt"
sha256sum "$BIN" >> "$ART/provenance.txt"
echo "LAANN_NATIVE_BUILD_PASS" | tee "$ART/build.status"

if [ ! -f "$RUNNER_DATA/dataset.json" ]; then
  if [ "${NH_LAANN_FETCH_DATA:-0}" != "1" ]; then
    echo "BUILD_ONLY: canonical SIFT1M not cached, fetching disabled" | tee "$ART/status.txt"
    exit 0
  fi
  python3 scripts/fetch_sift1m.py --cache "$CACHE/datasets" \
    --report "$ART/sift1m-dataset.json" > "$ART/fetch-data.log" 2>&1
fi
test -s "$RUNNER_DATA/dataset.json"
for name in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  test -s "$RUNNER_DATA/$name"
  ln -s "$RUNNER_DATA/$name" "$SRC/sift1M/$name"
done
ln -s "$INDEX_CACHE" "$SRC/sift1M/index"
BIN_CACHE="$CACHE/laann-native-sift1m/dataset-bin"
mkdir -p "$BIN_CACHE"
for name in sift_base.bin sift_query.bin sift_groundtruth.bin; do
  ln -s "$BIN_CACHE/$name" "$SRC/sift1M/$name"
done
shopt -s nullglob
EXISTING=("$INDEX_CACHE"/vamana_sift1M_*_LAANN_fsort.index)
shopt -u nullglob
if [ "${#EXISTING[@]}" -eq 0 ]; then
  sed -i 's/NAV_THREADS=16/NAV_THREADS=2/' "$SRC/sift1M/build_laann_sift1m.sh"
  nice -n 10 bash "$SRC/sift1M/build_laann_sift1m.sh" > "$ART/native-index-build.log" 2>&1
fi
shopt -s nullglob
EXISTING=("$INDEX_CACHE"/vamana_sift1M_*_LAANN_fsort.index)
shopt -u nullglob
test "${#EXISTING[@]}" -eq 1
PREFIX="${EXISTING[0]%.index}"
test -s "${PREFIX}_reorder_pq_compressed.bin"
test -s "${PREFIX}_pq_pivots.bin"
test -s "${PREFIX}_nav_graph.index"
GT="${PREFIX}_gt.bin"
test -s "$GT"
test -s "$BIN_CACHE/sift_query.bin"
sha256sum "${PREFIX}.index" "${PREFIX}_reorder_pq_compressed.bin" "$GT" >> "$ART/provenance.txt"

python3 scripts/laann_navhints/prepare_replay.py \
  --query "$BIN_CACHE/sift_query.bin" --reordered-gt "$GT" \
  --out "$WORK/replay" > "$ART/split.log"
mkdir -p "$ART/train" "$ART/results"
# Serial real-SSD activity via same global GeoIVF lock as other campaigns.
exec 9>"$CACHE/speed-device.lock"
flock -x 9
COMMON=(--data_type float --dist_fn l2 --index_path_prefix "$PREFIX" -K 10
        -W 5 -T 1 --use_laann --cache_ratio 0.2 --nav_L 100
        --use_look_ahead_search true --use_pipeline true)
TRACE="$ART/train/native-page-visits.txt"
rm -f "$TRACE"
NH_LAANN_MODE=trace NH_LAANN_TRACE_FILE="$TRACE" \
  nice -n 10 "$BIN" "${COMMON[@]}" --query_file "$WORK/replay/junction_train.bin" \
  --gt_file null -L 60 > "$ART/train/native-trace.log" 2>&1
test "$(wc -l < "$TRACE")" -eq 5000
python3 scripts/laann_navhints/build_directory.py \
  --trace "$TRACE" --reordered-pq "${PREFIX}_reorder_pq_compressed.bin" \
  --pivots "${PREFIX}_pq_pivots.bin" --output "$ART/train/page-junction-regions.txt" \
  > "$ART/train/directory-build.log" 2>&1
for L in 20 30 40 60 80 100 160; do
  for ARM in baseline junction core; do
    mkdir -p "$ART/results/$ARM"
    MODE=off
    if [ "$ARM" = junction ]; then MODE=junction; fi
    if [ "$ARM" = core ]; then MODE=core; fi
    OUT="$ART/results/$ARM/L$L.csv"
    NH_LAANN_MODE="$MODE" \
    NH_LAANN_DIRECTORY="$ART/train/page-junction-regions.txt" \
    NH_LAANN_HISTORY=2048 NH_LAANN_STARTS=8 NH_LAANN_NPROBE=8 \
    NH_LAANN_RESULTS="$OUT" \
      /usr/bin/time -v nice -n 10 "$BIN" "${COMMON[@]}" \
      --query_file "$WORK/replay/causal_replay.bin" \
      --gt_file "$WORK/replay/causal_replay_native_gt.bin" \
      -L "$L" > "$ART/results/$ARM/L$L.log" 2>&1
    test "$(wc -l < "$OUT")" -eq 5001
  done
done
flock -u 9
python3 scripts/laann_navhints/derive_results.py \
  --truth "$WORK/replay/causal_replay_native_gt.bin" \
  --results-root "$ART/results" --output "$ART/laann-matched-recall.json" \
  > "$ART/summary.log" 2>&1
echo "NH_LAANN_REAL_SSD_PASS" | tee "$ART/status.txt"
