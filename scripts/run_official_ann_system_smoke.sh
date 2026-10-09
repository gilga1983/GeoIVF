#!/usr/bin/env bash
# Validate that the *official* Starling and GPS/Gorgeous sources can build on
# our own self-hosted runner. No SSD benchmark or graph-index construction.
set -euo pipefail

ART="$PWD/artifacts/official-ann-system-smoke"
mkdir -p "$ART"
date -Is | tee "$ART/date.txt"
{ uname -a; lscpu; free -h; df -h; } > "$ART/host.txt"
git rev-parse HEAD > "$ART/geoivf-head.txt"
for cmd in git cmake make c++ python3 flock; do
  command -v "$cmd" || { echo "Missing command: $cmd"; exit 2; }
done
{ cmake --version; c++ --version; } > "$ART/toolchain.txt" 2>&1

python3 - <<'PY' | tee "$ART/dataset-readiness.txt"
from pathlib import Path
import json, os, struct
root=Path.home()/".cache/geoivf/public-eval"
for dataset in ("bigann-10M","text2image-10M"):
    mf=root/dataset/"data/dataset.manifest.json"
    print(f"DATASET {dataset} manifest={mf} exists={mf.is_file()}")
    if not mf.is_file():
        print("  Not staged locally: paper comparisons need the public causal dataset first.")
        continue
    m=json.loads(mf.read_text())
    for key in ("base","train5000","heldout5000","heldout5000_gt"):
        path=Path(m["files"][key])
        exists=path.is_file()
        print(f"  {key}: exists={exists} bytes={path.stat().st_size if exists else '-'} path={path}")
        if not exists:
            raise SystemExit("Missing staged public data")
        with path.open("rb") as f:
            h=f.read(8)
        rows,dim=struct.unpack("<II",h)
        print(f"    header=({rows},{dim})")
    print(f"  data_type={m['data_type']} dim={m['dim']} metric={m['metric']}")
PY

LOCK="$HOME/.cache/geoivf/speed-device.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
echo "Waiting for shared SSD/CPU qualification lock" | tee "$ART/lock.txt"
flock 9
echo "Acquired exclusive qualification lock $(date -Is)" | tee -a "$ART/lock.txt"

WORK="$RUNNER_TEMP/official-ann-smoke-$GITHUB_RUN_ID"
mkdir -p "$WORK"
echo "$WORK" > "$ART/workdir.txt"

verify_one() {
  local system="$1" url="$2" revision="$3"
  local root="$WORK/$system"
  echo "Cloning author source: $system $url" | tee "$ART/$system-status.txt"
  git clone --filter=blob:none "$url" "$root" \
    >"$ART/$system-clone.log" 2>&1
  git -C "$root" fetch --depth 1 origin "$revision" >>"$ART/$system-clone.log" 2>&1
  git -C "$root" checkout --detach "$revision" >>"$ART/$system-clone.log" 2>&1
  git -C "$root" submodule update --init --recursive --jobs 2 >>"$ART/$system-clone.log" 2>&1
  test "$(git -C "$root" rev-parse HEAD)" = "$revision"
  git -C "$root" submodule status > "$ART/$system-submodules.txt"
  echo "Source and submodule revisions pinned" | tee -a "$ART/$system-status.txt"

  # Build the native author's binaries. No edits to their code.
  cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release \
    >"$ART/$system-configure.log" 2>&1
  cmake --build "$root/build" --target build_disk_index search_disk_index --parallel 4 \
    >"$ART/$system-build.log" 2>&1
  for binary in build_disk_index search_disk_index; do
    local found="$root/build/tests/$binary"
    test -x "$found" || { echo "Missing $found" >&2; exit 5; }
    echo "$system native binary compiled: $found" | tee -a "$ART/$system-status.txt"
    file "$found" >> "$ART/$system-status.txt"
  done
  echo "SOURCE_BUILD_SUCCESS system=$system sha=$revision" | tee -a "$ART/$system-status.txt"
}

verify_one "starling" "https://github.com/zilliztech/starling.git" \
  "17dc3e8a011533a62374445f53963e951b72883a"
verify_one "gorgeous" "https://github.com/yinpeiqi/Gorgeous.git" \
  "04c8c27d77a19e7751748d4e0ac4ecd2f0fb3e8b"

echo "OFFICIAL_SOTA_BUILD_SMOKE_SUCCESS" | tee "$ART/result.txt"
