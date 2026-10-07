#!/usr/bin/env python3
"""Compose causal hub learning with online PQ-scored SkipDup value-cache seeds.

Apply after patch_diskann_online_hub_learning.py, itself applied after the
five-variant vertex-NavHints patch. All query routing is executed by the
unchanged DiskANN fixed-L frontier.

Both the online hub list and the 512-ID cache start empty and evolve from
completed searches. The cache stores only IDs, inserts rank-1 successes using
SkipDup/FIFO, and scores candidates with DiskANN's already-built PQ query LUT.

Hub lists are currently a causal in-memory *logical page overlay*: the page
read remains a prerequisite to offering its winners. Physical disk writes are
modeled and counted, not yet executed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    online_hub_winners: Option<&'a HashMap<u32, Vec<u32>>>,
}
""",
        """    online_hub_winners: Option<&'a HashMap<u32, Vec<u32>>>,
    value_cache_ids: Option<&'a [u32]>,
}
""",
        "combined HintIvfSearch state",
    )

    old_ctor = """            online_hub_winners: None,
        };
"""
    if s.count(old_ctor) < 2:
        raise RuntimeError("ordinary Hint-IVF initializers missing")
    s = s.replace(
        old_ctor,
        """            online_hub_winners: None,
            value_cache_ids: None,
        };
""",
    )

    emit_old = """            self.io_tracker.selected_hint_start.store(
                winner.1 as usize,
                std::sync::atomic::Ordering::Relaxed,
            );
            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
"""
    emit_new = r'''            self.io_tracker.selected_hint_start.store(
                winner.1 as usize,
                std::sync::atomic::Ordering::Relaxed,
            );

            let mut emitted = 1usize;
            if let Some(cache_ids) = ivf.value_cache_ids {
                if !cache_ids.is_empty() {
                    let mut best_cache: Option<(f32, u32)> = None;
                    self.pq_distances(cache_ids, |distance, id| {
                        let better = best_cache.is_none_or(|current| {
                            distance
                                .total_cmp(&current.0)
                                .then_with(|| id.cmp(&current.1))
                                .is_lt()
                        });
                        if better {
                            best_cache = Some((distance, id));
                        }
                    })?;
                    routing_cmps = routing_cmps.checked_add(cache_ids.len()).ok_or_else(|| {
                        diskann_error!(ErrorKind::IndexError, "combined cache comparison overflow")
                    })?;

                    if let Some((distance, id)) = best_cache {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
                }
            }

            self.io_tracker.routing_comparisons.fetch_add(
                routing_cmps.saturating_sub(emitted),
                std::sync::atomic::Ordering::Relaxed,
            );
            f(winner.1, winner.0);
            return Ok(());
'''
    s = once(s, emit_old, emit_new, "combined online cached start")

    s = once(
        s,
        """        online_hub_winners: &HashMap<u32, Vec<u32>>,
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32)> {
""",
        """        online_hub_winners: &HashMap<u32, Vec<u32>>,
        value_cache_ids: &[u32],
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32)> {
""",
        "combined public method arg",
    )
    s = once(
        s,
        """            online_hub_winners: Some(online_hub_winners),
        };
""",
        """            online_hub_winners: Some(online_hub_winners),
            value_cache_ids: Some(value_cache_ids),
        };
""",
        "combined public method cache view",
    )

    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()
    s = once(
        s,
        """    if online_hub_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
""",
        """    let combined_value_cache_capacity = std::env::var("DISKANN_COMBINED_VALUE_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if combined_value_cache_capacity > 4096 {
        anyhow::bail!("combined cache capacity limit for this screen is 4096");
    }
    if online_hub_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
""",
        "combined cache configuration",
    )

    s = once(
        s,
        """            let mut batch_writes = 0usize;

            for qi in 0..num_queries {
                let q = queries.row(qi);
""",
        """            let mut batch_writes = 0usize;

            let mut cache_ids = Vec::<u32>::with_capacity(combined_value_cache_capacity);
            let mut cache_members = HashSet::<u32>::with_capacity(combined_value_cache_capacity);
            let mut cache_next = 0usize;
            let mut cache_inserts = 0usize;
            let mut cache_skips = 0usize;
            let mut cache_evictions = 0usize;

            for qi in 0..num_queries {
                let query_timer = Instant::now();
                let q = queries.row(qi);
""",
        "combined cache replay state",
    )

    s = once(
        s,
        """                        variant,
                        &online_hubs,
                    )?;
""",
        """                        variant,
                        &online_hubs,
                        &cache_ids,
                    )?;
""",
        "combined cache argument in replay",
    )

    s = once(
        s,
        """                statistics_vec[qi] = search_result.stats.query_statistics;

                if online_hub_capacity > 0 && base_count > 0 {
""",
        """                if online_hub_capacity > 0 && base_count > 0 {
""",
        "defer query timing until after online updates",
    )

    s = once(
        s,
        """                }
            }

            let dirty_pages = pending.values().filter(|x| **x > 0).count();
""",
        """                }

                if combined_value_cache_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    if cache_members.contains(&winner) {
                        cache_skips += 1;
                    } else {
                        if cache_ids.len() < combined_value_cache_capacity {
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                        } else {
                            let victim = cache_ids[cache_next];
                            cache_members.remove(&victim);
                            cache_ids[cache_next] = winner;
                            cache_members.insert(winner);
                            cache_next = (cache_next + 1) % combined_value_cache_capacity;
                            cache_evictions += 1;
                        }
                        cache_inserts += 1;
                    }
                }

                // Account for the real cache lookup and both online update
                // paths as part of request latency. This includes bookkeeping
                // but deliberately excludes physical page writes not yet made.
                let elapsed_us = query_timer.elapsed().as_micros();
                let stats = &mut statistics_vec[qi];
                *stats = search_result.stats.query_statistics;
                let search_us = stats.total_execution_time_us;
                stats.cpu_time_us = stats
                    .cpu_time_us
                    .saturating_add(elapsed_us.saturating_sub(search_us));
                stats.total_execution_time_us = elapsed_us;
            }

            let dirty_pages = pending.values().filter(|x| **x > 0).count();
""",
        "combined cache updates and full query timing",
    )

    s = once(
        s,
        """                batch_writes + dirty_pages,
            );
""",
        """                batch_writes + dirty_pages,
            );
            eprintln!(
                "ONLINE_VALUE_STATS L={} capacity={} occupancy={} inserts={} duplicate_skips={} evictions={}",
                l,
                combined_value_cache_capacity,
                cache_ids.len(),
                cache_inserts,
                cache_skips,
                cache_evictions,
            );
""",
        "combined per-L cache accounting",
    )

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    provider = root / "diskann-disk/src/search/provider/disk_provider.rs"
    benchmark = root / "diskann-benchmark/src/disk_index/search.rs"
    if not provider.is_file() or not benchmark.is_file():
        raise SystemExit("unexpected DiskANN checkout layout")
    patch_provider(provider)
    patch_benchmark(benchmark)
    print("patched causal online hubs + online resident-PQ value cache")


if __name__ == "__main__":
    main()
