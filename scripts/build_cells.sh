#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
${CXX:-g++} -O3 -march=native -std=c++17 -fPIC -shared -Wall -Wextra -Werror \
  -fno-fast-math -ffp-contract=off native/cell_bounds.cpp -o build/libgeoivf_cells.so
