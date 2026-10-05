#!/usr/bin/env bash
set -euxo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.2.6

ROOT_ART="$PWD/artifacts/public-eval"
rm -rf "$ROOT_ART"
mkdir -p "$ROOT_ART"

# Run sequentially on the shared host so index construction for one dataset
# cannot distort timed measurements for the other.
bash scripts/run_public_dataset_one.sh text2image-10M "$ROOT_ART/text2image-10M"
bash scripts/run_public_dataset_one.sh bigann-10M "$ROOT_ART/bigann-10M"
