#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/build_ramfirst.sh "$@"
${CXX:-g++} -O3 -march=native -std=c++17 -fPIC -shared -Wall -Wextra -Werror \
  -fno-fast-math -ffp-contract=off native/rotation_rank.cpp -o build/libgeoivf_rotation.so
