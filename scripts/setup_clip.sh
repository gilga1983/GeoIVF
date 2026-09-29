#!/usr/bin/env bash
# Build the released implementation without changing its search/training code.
# Ubuntu/Debian BLAS packages are downloaded/extracted privately, never installed.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
rev=7f4fc84edffede0aa21fae6131ec7391ce99ab6f
src="$root/third_party/CLIP"
mkdir -p "$root/artifacts/upstream"
if [[ ! -d "$src/.git" ]]; then
    git clone --filter=blob:none https://github.com/SongYitong826/CLIP.git "$src"
fi
git -C "$src" checkout --detach "$rev"
git -C "$src" submodule update --init --recursive --depth=1
[[ "$(git -C "$src" rev-parse HEAD)" == "$rev" ]]
{ git -C "$src" rev-parse HEAD; git -C "$src" submodule status; } > "$root/artifacts/upstream/clip-revisions.txt"
# Explicit local BLAS keeps this setup independent of sudo and host package state.
private="$root/third_party/blas"
mkdir -p "$private/packages" "$private/root"
(
    cd "$private/packages"
    apt-get download libopenblas0-pthread libopenblas-pthread-dev libgfortran5 libquadmath0
    for deb in ./*.deb; do
        dpkg-deb -x "$deb" "$private/root"
        dpkg-deb -f "$deb" Package Version
        sha256sum "$deb"
    done
) > "$root/artifacts/upstream/blas-packages.txt"
triplet="$(gcc -dumpmachine)"
libdir="$private/root/usr/lib/$triplet"
blas="$libdir/openblas-pthread/libopenblas.so.0"
[[ -f "$blas" ]]
export LD_LIBRARY_PATH="$libdir:$libdir/openblas-pthread${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S "$src" -B "$root/build/clip" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DBLAS_LIBRARIES="$blas" -DLAPACK_LIBRARIES="$blas" \
    -DCMAKE_BUILD_RPATH="$libdir;$libdir/openblas-pthread"
cmake --build "$root/build/clip" --target test_ivf_clip test_hivf_clip test_ivfflat -j2
printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH" > "$root/build/clip-env.sh"
printf 'Upstream binaries built. No canonical CLIP performance claim yet.\n' > "$root/artifacts/upstream/clip-build-status.txt"
