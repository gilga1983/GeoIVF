#!/usr/bin/env bash
# Authors' original CatapultDB Rust search over their C++ DiskANN/Vamana graph.
# Causal query order, frozen Coveo split. IN MEMORY: no SSD read comparisons.
set -euo pipefail
set -x
ART="$PWD/artifacts/catapult-author-coveo"
WORK="$RUNNER_TEMP/catapult-author-coveo-$GITHUB_RUN_ID"
LOCK="$HOME/.cache/geoivf/speed-device.lock"
MANIFEST="$HOME/.cache/geoivf/coveo-real-demand-v1/split/dataset.manifest.json"
AUTHOR_RUST="a16eddd34b4339db5ec86e292470ce7929179bc3"
AUTHOR_CPP="c1dbaecce5e7e02d02ad1493660c8048d965005d"
TOOLCHAIN="nightly-2026-09-15"
mkdir -p "$ART" "$WORK" "$(dirname "$LOCK")"
test -s "$MANIFEST"
git rev-parse HEAD > "$ART/geoivf-sha.txt"
{ date -Is; uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
exec 9>"$LOCK"
echo "Waiting for exclusive CPU and SSD lock" | tee "$ART/lock.log"
flock 9
echo "Acquired lock $(date -Is)" | tee -a "$ART/lock.log"

git clone --filter=blob:none https://github.com/sacs-epfl/catapulted-diskann.git "$WORK/cpp"
git -C "$WORK/cpp" checkout --detach "$AUTHOR_CPP"
git -C "$WORK/cpp" submodule update --init --recursive --jobs 2
test "$(git -C "$WORK/cpp" rev-parse HEAD)" = "$AUTHOR_CPP"
printf '%s\n' "$AUTHOR_CPP" > "$ART/author-cpp-sha.txt"

git clone --filter=blob:none https://github.com/MRandl/catapult-db.git "$WORK/rust"
git -C "$WORK/rust" checkout --detach "$AUTHOR_RUST"
test "$(git -C "$WORK/rust" rev-parse HEAD)" = "$AUTHOR_RUST"
printf '%s\n' "$AUTHOR_RUST" > "$ART/author-rust-sha.txt"

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.6
"$WORK/venv/bin/python" scripts/prepare_catapult_original_coveo.py "$MANIFEST" "$WORK/prepared" "$ART"

cmake -S "$WORK/cpp" -B "$WORK/cpp/build" -DCMAKE_BUILD_TYPE=Release > "$ART/cpp-cmake.log" 2>&1
cmake --build "$WORK/cpp/build" --parallel 4 --target build_memory_index > "$ART/cpp-build.log" 2>&1
test -x "$WORK/cpp/build/apps/build_memory_index"
mkdir -p "$WORK/index"
"$WORK/cpp/build/apps/build_memory_index" --data_type float --dist_fn l2 --data_path "$WORK/prepared/coveo-64d-base.fbin" --index_path_prefix "$WORK/index/coveo-vamana" --max_degree 48 --Lbuild 128 --alpha 1.2 --num_threads 4 > "$ART/index-build.log" 2>&1
python3 scripts/verify_catapult_author_graph.py "$WORK/index/coveo-vamana" "$WORK/index/coveo-vamana.data"

mkdir -p "$WORK/rust/src/bin"
cp scripts/catapult_original_coveo_eval.rs "$WORK/rust/src/bin/navhints_author_coveo_eval.rs"
export RUSTUP_HOME="$HOME/.cache/geoivf/catapult-authors-rustup"
export CARGO_HOME="$HOME/.cache/geoivf/catapult-authors-cargo"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME"
if ! command -v rustup >/dev/null 2>&1; then
  curl -fsSL --retry 3 https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain "$TOOLCHAIN"
fi
rustup toolchain install "$TOOLCHAIN" --profile minimal
(
 cd "$WORK/rust"
 cargo "+$TOOLCHAIN" build --release --locked --bin navhints_author_coveo_eval > "$ART/rust-build.log" 2>&1
)
BIN="$WORK/rust/target/release/navhints_author_coveo_eval"
test -x "$BIN"

for rep in 0 1 2; do
 case "$rep" in
    0) seed=42; order="vanilla catapult" ;;
    1) seed=123; order="catapult vanilla" ;;
    2) seed=2026; order="vanilla catapult" ;;
 esac
 for L in 12 20 40 80 160; do
  for mode in $order; do
   echo "CATAPULT_AUTHOR_COVEO_START rep=$rep L=$L mode=$mode" | tee -a "$ART/measurements.log"
   "$BIN" "$WORK/index/coveo-vamana" "$WORK/index/coveo-vamana.data" "$WORK/prepared/coveo-64d-replay.npy" "$WORK/prepared/replay.gt" "$mode" "$L" "$seed" "$ART/rep$rep-L$L-$mode.json" > "$ART/rep$rep-L$L-$mode.log" 2>&1
   echo "CATAPULT_AUTHOR_COVEO_DONE rep=$rep L=$L mode=$mode" | tee -a "$ART/measurements.log"
  done
 done
done
python3 scripts/summarize_catapult_original_coveo.py "$ART"
printf '%s\n' "CATAPULT_AUTHOR_ORIGINAL_COVEO_VALIDATED_SUCCESS" | tee "$ART/result.txt"
