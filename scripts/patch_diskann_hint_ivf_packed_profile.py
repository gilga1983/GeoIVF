#!/usr/bin/env python3
"""Profile the current packed-direct Hint-IVF path by stage.

Profiling-only build. Installs the canonical packed-direct implementation and
adds per-query timers for:
  * coarse packed representative scoring + top-k maintenance;
  * fine selected-bucket scoring + winner selection.

Headline throughput measurements never use this instrumented binary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patch_diskann_start_points import PINNED, once, patch_provider
from patch_diskann_paper_catapult import patch_provider_medoid
from patch_diskann_waypoint_cache import patch_provider_waypoint
from patch_diskann_global_starts import patch_benchmark_global
from patch_diskann_hint_ivf_packed_direct import (
    expose_pq_batch_lookup,
    patch_provider_hint_ivf_fast,
    patch_benchmark_hint_ivf_fast,
)


def patch_statistics(path: Path) -> None:
    s = path.read_text()
    old = """    /// Time spent in query preprocessing for the PQ in microseconds.
    pub query_pq_preprocess_time_us: u128,

    /// Total number of IO operations issued.
"""
    new = """    /// Time spent in query preprocessing for the PQ in microseconds.
    pub query_pq_preprocess_time_us: u128,

    /// Profiling-only time spent scoring/selecting Hint-IVF representatives.
    pub hint_ivf_coarse_time_us: u128,

    /// Profiling-only time spent scoring selected Hint-IVF buckets.
    pub hint_ivf_fine_time_us: u128,

    /// Total number of IO operations issued.
"""
    s = once(s, old, new, "packed profile QueryStatistics fields")
    path.write_text(s)


def patch_provider_profile(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """struct IOTracker {
    io_time_us: AtomicU64,
    preprocess_time_us: AtomicU64,
    io_count: AtomicUsize,
    routing_comparisons: AtomicUsize,
}
""",
        """struct IOTracker {
    io_time_us: AtomicU64,
    preprocess_time_us: AtomicU64,
    io_count: AtomicUsize,
    routing_comparisons: AtomicUsize,
    routing_coarse_time_us: AtomicU64,
    routing_fine_time_us: AtomicU64,
}
""",
        "packed profile IOTracker fields",
    )
    s = once(
        s,
        """            io_count: AtomicUsize::new(0),
            routing_comparisons: AtomicUsize::new(0),
        }
""",
        """            io_count: AtomicUsize::new(0),
            routing_comparisons: AtomicUsize::new(0),
            routing_coarse_time_us: AtomicU64::new(0),
            routing_fine_time_us: AtomicU64::new(0),
        }
""",
        "packed profile IOTracker init",
    )

    old = """            self.pq_distances_packed(
                ivf.coarse_local_ids,
                ivf.coarse_pq_codes,
"""
    new = """            let coarse_timer = Instant::now();
            self.pq_distances_packed(
                ivf.coarse_local_ids,
                ivf.coarse_pq_codes,
"""
    s = once(s, old, new, "packed profile coarse timer start")

    old = """            },
            )?;

            // Preserve the same canonical fine winner, but gather selected
"""
    new = """            },
            )?;
            IOTracker::add_time(
                &self.io_tracker.routing_coarse_time_us,
                coarse_timer.elapsed().as_micros() as u64,
            );

            // Preserve the same canonical fine winner, but gather selected
"""
    s = once(s, old, new, "packed profile coarse timer end")

    old = """            let mut winner: Option<(f32, u32)> = None;
            let mut routing_cmps = ivf.medoid_ids.len();
"""
    new = """            let fine_timer = Instant::now();
            let mut winner: Option<(f32, u32)> = None;
            let mut routing_cmps = ivf.medoid_ids.len();
"""
    s = once(s, old, new, "packed profile fine timer start")

    old = """            let winner = winner.ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF selected no start")
            })?;

            // The graph search counts the emitted start once. Record the other
"""
    new = """            let winner = winner.ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF selected no start")
            })?;
            IOTracker::add_time(
                &self.io_tracker.routing_fine_time_us,
                fine_timer.elapsed().as_micros() as u64,
            );

            // The graph search counts the emitted start once. Record the other
"""
    s = once(s, old, new, "packed profile fine timer end")

    old = """        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.cpu_time_us = query_stats
"""
    new = """        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.hint_ivf_coarse_time_us =
            IOTracker::time(&io_tracker.routing_coarse_time_us) as u128;
        query_stats.hint_ivf_fine_time_us =
            IOTracker::time(&io_tracker.routing_fine_time_us) as u128;
        query_stats.cpu_time_us = query_stats
"""
    s = once(s, old, new, "packed profile QueryStatistics assignment")
    path.write_text(s)


def patch_benchmark_profile(path: Path) -> None:
    s = path.read_text()
    s = once(
        s,
        """    pub(super) mean_pq_preprocess_time: f64,
    pub(super) mean_comparisons: f64,
""",
        """    pub(super) mean_pq_preprocess_time: f64,
    pub(super) mean_hint_ivf_coarse_time: f64,
    pub(super) mean_hint_ivf_fine_time: f64,
    pub(super) mean_comparisons: f64,
""",
        "packed profile result fields",
    )
    s = once(
        s,
        """            mean_pq_preprocess_time: statistics::get_mean_stats(statistics, |stats| {
                stats.query_pq_preprocess_time_us as f64
            }),
            mean_comparisons: statistics::get_mean_stats(statistics, |stats| {
""",
        """            mean_pq_preprocess_time: statistics::get_mean_stats(statistics, |stats| {
                stats.query_pq_preprocess_time_us as f64
            }),
            mean_hint_ivf_coarse_time: statistics::get_mean_stats(statistics, |stats| {
                stats.hint_ivf_coarse_time_us as f64
            }),
            mean_hint_ivf_fine_time: statistics::get_mean_stats(statistics, |stats| {
                stats.hint_ivf_fine_time_us as f64
            }),
            mean_comparisons: statistics::get_mean_stats(statistics, |stats| {
""",
        "packed profile result aggregation",
    )
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()

    provider = root / "diskann-disk/src/search/provider/disk_provider.rs"
    statistics = root / "diskann-disk/src/utils/statistics.rs"
    benchmark = root / "diskann-benchmark/src/disk_index/search.rs"
    if not provider.is_file() or not statistics.is_file() or not benchmark.is_file():
        raise SystemExit("unexpected DiskANN checkout layout")

    expose_pq_batch_lookup(root)
    patch_provider(provider)
    patch_provider_medoid(provider)
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf_fast(provider)
    patch_benchmark_hint_ivf_fast(benchmark)

    patch_statistics(statistics)
    patch_provider_profile(provider)
    patch_benchmark_profile(benchmark)
    print(f"patched DiskANN {PINNED} with packed-direct Hint-IVF stage profiling")


if __name__ == "__main__":
    main()
