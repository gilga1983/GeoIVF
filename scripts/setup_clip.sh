#!/usr/bin/env bash
# Build the released implementation without changing its search/training code.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
rev=7f4fc84edffede0aa21fae6131ec7391ce99ab6f
src="$root/third_party/CLIP"
mkdir -p "$root/artifacts/upstream"
if [[ ! -d "$src/.git" ]]; then git clone https://github.com/SongYitong826/CLIP.git "$src"; fi
git -C "$src" checkout --detach "$rev"
git -C "$src" submodule update --init --recursive
[[ "$(git -C "$src" rev-parse HEAD)" == "$rev" ]]
{ git -C "$src" rev-parse HEAD; git -C "$src" submodule status; } > "$root/artifacts/upstream/clip-revisions.txt"
cmake -S "$src" -B "$root/build/clip" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "$root/build/clip" --target test_ivf_clip test_hivf_clip test_ivfflat -j2
printf 'Upstream binaries built. No canonical CLIP performance claim yet.\n' > "$root/artifacts/upstream/clip-build-status.txt"
