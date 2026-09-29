#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/build_prepared.sh
${CXX:-g++} -O3 -march=native -std=c++17 -fPIC -shared -Wall -Wextra -Werror \
  -fno-fast-math -ffp-contract=off native/simd_topk.cpp -o build/libgeoivf_topk_simd.so
if [[ "${1:-}" == "--uring" ]]; then
  bash scripts/setup_uring.sh
  ${CXX:-g++} -O3 -std=c++17 -fPIC -shared -Wall -Wextra -Werror -DGEOIVF_URING \
    -Ithird_party/liburing/src/include native/rolling_reader.cpp \
    third_party/liburing/src/liburing.a -o build/libgeoivf_rolling.so
fi
