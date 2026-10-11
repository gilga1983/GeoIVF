#!/usr/bin/env bash
# Common host-wide guard for GeoIVF performance measurements.
#
# Usage:
#   source scripts/perf_guard.sh
#   geoivf_perf_lock
#
# Keep the returned FD open for the whole timed experiment. This is separate
# from speed-device.lock: the device lock protects SSD timing, while this guard
# prevents two new GeoIVF performance campaigns from sharing CPU/memory.
set -o pipefail

geoivf_perf_lock() {
  local lock_path="${HOME}/.cache/geoivf/perf-host.lock"
  mkdir -p "$(dirname "$lock_path")"
  exec 198>"$lock_path"
  flock -x 198
}
