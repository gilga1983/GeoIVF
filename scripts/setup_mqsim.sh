#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/third_party/MQSim}"
REV=51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16
URL=https://github.com/CMU-SAFARI/MQSim.git
if [[ ! -d "$DEST/.git" ]]; then
    if [[ -e "$DEST" ]]; then echo "Destination already exists: $DEST" >&2; exit 1; fi
    mkdir -p "$(dirname "$DEST")"
    git clone "$URL" "$DEST"
fi
[[ "$(git -C "$DEST" remote get-url origin)" == "$URL" ]] || { echo 'Unexpected origin' >&2; exit 1; }
[[ -z "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]] || { echo 'Modified MQSim source' >&2; exit 1; }
git -C "$DEST" fetch origin "$REV"
git -C "$DEST" checkout --detach "$REV"
make -C "$DEST" -j"${BUILD_JOBS:-2}"
printf 'MQSim revision: %s\n' "$(git -C "$DEST" rev-parse HEAD)"
