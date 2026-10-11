#!/usr/bin/env bash
# Source this file: source repro/activate.sh
# Sets local, non-destructive defaults and exposes one audited repro launcher.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source repro/activate.sh" >&2
  exit 2
fi

_NAVHINTS_THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export NAVHINTS_ROOT="$(cd -- "$_NAVHINTS_THIS_DIR/.." && pwd -P)"
export NAVHINTS_DATA_ROOT="$HOME/.cache/geoivf"
export NAVHINTS_PAPER_ROOT="${NAVHINTS_PAPER_ROOT:-$(cd -- "$NAVHINTS_ROOT/.." && pwd -P)/DemmandAwareANN}"
export RUNNER_TEMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
export GITHUB_RUN_ID="${GITHUB_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export GITHUB_RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

navhints() {
  python3 "$NAVHINTS_ROOT/repro/run.py" "$@"
}
navhints_doctor() {
  navhints doctor "$@"
}
navhints_paper() {
  navhints paper --paper-dir "$NAVHINTS_PAPER_ROOT" "$@"
}
echo "NavHints reproduction environment: $NAVHINTS_ROOT"
echo "  run: navhints list"
echo "  inspect: navhints doctor --mode pubmed"
echo "  replay: navhints run pubmed --execute (heavy, real SSD)"
