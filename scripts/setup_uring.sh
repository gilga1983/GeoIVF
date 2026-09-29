#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
rev=08468cc3830185c75f9e7edefd88aa01e5c2f8ab
src="$root/third_party/liburing"
if [[ ! -d "$src/.git" ]]; then
    git clone https://github.com/axboe/liburing.git "$src"
fi
git -C "$src" checkout --detach "$rev"
[[ "$(git -C "$src" rev-parse HEAD)" == "$rev" ]]
(cd "$src" && ./configure && make -C src -B -j2 CFLAGS='-O2 -fPIC -Wall')
mkdir -p "$root/build"
g++ -O3 -std=c++17 -Wall -Wextra -Werror -fPIC -DGEOIVF_URING \
    -I"$src/src/include" -shared "$root/native/reader.cpp" \
    "$src/src/liburing.a" -o "$root/build/libgeoivf_io.so"
printf '%s\n' "$rev"
