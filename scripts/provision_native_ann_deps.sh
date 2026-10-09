#!/usr/bin/env bash
# Provision pinned upstream C++ build dependencies into this job's scratch tree.
# Never runs apt install or sudo, and never changes the runner's system packages.
set -euo pipefail

ROOT="${1:?usage: provision_native_ann_deps.sh ROOT}"
mkdir -p "$ROOT/debs/partial" "$ROOT/sysroot"
for command in apt-get dpkg-deb awk; do
  command -v "$command" >/dev/null || { echo "Missing $command" >&2; exit 2; }
done

# Author projects link Intel MKL, Boost program_options, libaio and tcmalloc.
# The old upstream launchers assume these packages already exist globally.
packages=(libboost-dev libboost-program-options-dev libaio-dev libgoogle-perftools-dev libmkl-full-dev)
echo "Inspecting Ubuntu package dependencies (no root required)"
apt-get -s -o Debug::NoLocking=true --no-install-recommends install "${packages[@]}" > "$ROOT/apt-plan.txt" || {
  cat "$ROOT/apt-plan.txt" >&2
  echo "APT cannot resolve official upstream dependencies; verify Ubuntu universe/multiverse and apt indexes." >&2
  exit 3
}
mapfile -t needed < <(awk '$1 == "Inst" {print $2}' "$ROOT/apt-plan.txt" | sort -u)
if (("${#needed[@]}" > 0)); then
  echo "Downloading ${#needed[@]} missing package archives into job scratch (no system install)"
  (cd "$ROOT/debs" && apt-get download "${needed[@]}") >"$ROOT/download.log" 2>&1 || {
    tail -n 120 "$ROOT/download.log" >&2
    exit 4
  }
  for deb in "$ROOT"/debs/*.deb; do
    test -f "$deb" || continue
    dpkg-deb -x "$deb" "$ROOT/sysroot"
  done
fi

# APT can omit packages already installed globally. The original system's
# include/library paths remain on the C++ compiler's search path in that case.
SYS="$ROOT/sysroot/usr"
LIB="$SYS/lib/x86_64-linux-gnu"
MKL="$LIB/mkl"
test -d "$SYS" || mkdir -p "$SYS"
{
  printf 'export BOOST_ROOT=%q\n' "$SYS"
  printf 'export BOOST_INCLUDEDIR=%q\n' "$SYS/include"
  printf 'export BOOST_LIBRARYDIR=%q\n' "$LIB"
  printf 'export CMAKE_PREFIX_PATH=%q:${CMAKE_PREFIX_PATH:-}\n' "$SYS"
  printf 'export CPLUS_INCLUDE_PATH=%q:%q:${CPLUS_INCLUDE_PATH:-}\n' "$SYS/include" "$SYS/include/mkl"
  printf 'export C_INCLUDE_PATH=%q:%q:${C_INCLUDE_PATH:-}\n' "$SYS/include" "$SYS/include/mkl"
  printf 'export LIBRARY_PATH=%q:%q:%q:${LIBRARY_PATH:-}\n' "$LIB" "$MKL" "$SYS/lib"
  printf 'export LD_LIBRARY_PATH=%q:%q:%q:${LD_LIBRARY_PATH:-}\n' "$LIB" "$MKL" "$SYS/lib"
  printf 'export LDFLAGS=-L%q\ -L%q\ -L%q\ ${LDFLAGS:-}\n' "$LIB" "$MKL" "$SYS/lib"
} > "$ROOT/native-deps.env"

echo "Upstream dependencies staged: $(du -sh "$ROOT/sysroot" | cut -f1) scratch"
echo "Sourcing $ROOT/native-deps.env provides Boost, MKL, libaio and tcmalloc search paths"
