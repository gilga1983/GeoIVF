#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
rev=7f4fc84edffede0aa21fae6131ec7391ce99ab6f
src="$root/third_party/CLIP"
mkdir -p artifacts/external-ivf
if [[ ! -d "$src/.git" ]]; then git clone --filter=blob:none https://github.com/SongYitong826/CLIP.git "$src"; fi
git -C "$src" checkout --detach "$rev"
git -C "$src" submodule update --init --recursive --depth=1
{ git -C "$src" rev-parse HEAD; git -C "$src" submodule status; } > artifacts/external-ivf/upstream-revisions.txt
private="$root/third_party/blas"
mkdir -p "$private/packages" "$private/root"
(cd "$private/packages"; apt-get download libopenblas0-pthread libopenblas-pthread-dev libgfortran5 libquadmath0; for deb in ./*.deb; do dpkg-deb -x "$deb" "$private/root"; dpkg-deb -f "$deb" Package Version; sha256sum "$deb"; done) > artifacts/external-ivf/blas-provenance.txt
triplet="$(gcc -dumpmachine)";libdir="$private/root/usr/lib/$triplet";blas="$libdir/openblas-pthread/libopenblas.so.0"
[[ -f "$blas" ]]
export LD_LIBRARY_PATH="$libdir:$libdir/openblas-pthread${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S benchmarks/upstream-clip -B build/external-clip -DCMAKE_BUILD_TYPE=Release \
  -DCLIP_SOURCE="$src" -DBUILD_TESTING=OFF -DBLAS_LIBRARIES="$blas" -DLAPACK_LIBRARIES="$blas" \
  -DCMAKE_BUILD_RPATH="$libdir;$libdir/openblas-pthread"
cmake --build build/external-clip --target geoivf_upstream -j4
printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH" > build/external-env.sh
# The bridge and build glue are ours. Every upstream source stays unmodified.
git -C "$src" diff --exit-code
