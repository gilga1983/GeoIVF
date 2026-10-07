#!/usr/bin/env python3
"""Persist query-relevant subsets of the EXISTING 512-ID winner cache.

Apply after:
  patch_diskann_vertex_navhints.py --variants 5
  patch_diskann_online_hub_learning.py
  patch_diskann_online_combined.py

Every successful request still inserts its rank-1 winner into the SAME
SkipDup/FIFO 512-ID cache. We reuse the cache's resident-PQ scan to remember
the ten best cached winners for the current query. Every X completed queries,
one hub-page snapshot is published for the query's selected 16K start hub:
rank-1 current winner, previously published hub winners, and then best distinct cached winners to fill the remaining slots.

No second cache, query key, per-hub accumulation buffer, or second PQ scan.
The hub page overlay is causally visible to *future* queries only, and still
requires that hub's native page expansion. This first experiment models writes:
one snapshot update counts as one physical 4-KiB page write, but it does not
yet mutate the index on SSD.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    selected_hint_start: AtomicUsize,
}
""",
        """    selected_hint_start: AtomicUsize,
    snapshot_candidates: std::sync::Mutex<Vec<u32>>,
}
""",
        "snapshot per-query IOTracker field",
    )
    s = once(
        s,
        """            selected_hint_start: AtomicUsize::new(usize::MAX),
        }
""",
        """            selected_hint_start: AtomicUsize::new(usize::MAX),
            snapshot_candidates: std::sync::Mutex::new(Vec::new()),
        }
""",
        "snapshot per-query IOTracker init",
    )

    s = once(
        s,
        """    value_cache_ids: Option<&'a [u32]>,
}
""",
        """    value_cache_ids: Option<&'a [u32]>,
    snapshot_candidates_limit: usize,
}
""",
        "snapshot HintIvfSearch field",
    )

    # Existing non-online Hint-IVF methods still initialize the view with no
    # online cache and no snapshots.
    init_old = """            value_cache_ids: None,
        };
"""
    if s.count(init_old) < 2:
        raise RuntimeError(f"expected at least two HintIvfSearch default initializers, got {s.count(init_old)}")
    s = s.replace(
        init_old,
        """            value_cache_ids: None,
            snapshot_candidates_limit: 0,
        };
""",
    )

    s = once(
        s,
        """        value_cache_ids: &[u32],
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32)> {
""",
        """        value_cache_ids: &[u32],
        snapshot_candidates_limit: usize,
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32, Vec<u32>)> {
""",
        "snapshot public search arguments",
    )

    s = once(
        s,
        """            value_cache_ids: Some(value_cache_ids),
        };
""",
        """            value_cache_ids: Some(value_cache_ids),
            snapshot_candidates_limit,
        };
""",
        "snapshot online HintIvfSearch constructor",
    )

    old_scan = r'''                    let mut best_cache: Option<(f32, u32)> = None;
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
'''
    new_scan = r'''                    let mut best_cache: Option<(f32, u32)> = None;
                    let limit = ivf.snapshot_candidates_limit.min(10);
                    let mut ranked = Vec::<(f32, u32)>::with_capacity(limit);
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

                        // Only snapshot arms maintain an ordered top-10. The
                        // exact PQ distances are already computed for lookup.
                        if limit > 0 {
                            let pos = ranked.partition_point(|(d, x)| {
                                d.total_cmp(&distance)
                                    .then_with(|| x.cmp(&id))
                                    .is_lt()
                            });
                            if pos < limit {
                                ranked.insert(pos, (distance, id));
                                if ranked.len() > limit {
                                    ranked.pop();
                                }
                            }
                        }
                    })?;
                    if limit > 0 {
                        let mut selected = self.io_tracker.snapshot_candidates
                            .lock()
                            .map_err(|_| diskann_error!(
                                ErrorKind::IndexError, "poisoned snapshot mutex"
                            ))?;
                        selected.extend(ranked.into_iter().map(|(_, id)| id));
                    }
                    routing_cmps = routing_cmps.checked_add(cache_ids.len()).ok_or_else(|| {
'''
    s = once(s, old_scan, new_scan, "reuse cache PQ scan for top10 snapshot")

    old_return = """        Ok((search_result, selected_hub))
    }
"""
    new_return = """        let ranked = io_tracker.snapshot_candidates
            .lock()
            .map_err(|_| diskann_error!(
                ErrorKind::IndexError, "poisoned snapshot mutex"
            ))?
            .clone();
        Ok((search_result, selected_hub, ranked))
    }
"""
    s = once(s, old_return, new_return, "return selected cached values")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()
    s = once(
        s,
        """    if online_hub_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
""",
        """    // Zero preserves the original independent online-hub learner.
    // Positive values select one cached-winner subset for publication every
    // X completed requests, with no additional write cache.
    let winner_snapshot_cadence = std::env::var("DISKANN_WINNER_SNAPSHOT_CADENCE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if winner_snapshot_cadence > 0
        && (!online_hub_replay
            || combined_value_cache_capacity == 0
            || online_hub_capacity == 0)
    {
        anyhow::bail!("winner snapshots require online replay, a value cache, and hub slots");
    }

    if online_hub_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
""",
        "snapshot cadence configuration",
    )

    s = once(
        s,
        """            let mut cache_evictions = 0usize;

            for qi in 0..num_queries {
""",
        """            let mut cache_evictions = 0usize;

            let mut snapshot_writes = 0usize;
            let mut snapshot_nochange = 0usize;
            let mut snapshot_overwrites = 0usize;
            let mut snapshot_unique_pages = HashSet::<u32>::new();
            let mut snapshot_total_ids = 0usize;
            let mut snapshot_written_warm = 0usize;
            let mut snapshot_written_eval = 0usize;

            for qi in 0..num_queries {
""",
        "snapshot accounting state",
    )

    s = once(
        s,
        """                let (search_result, selected_hub) =
                    searcher.search_with_vertex_hint_ivf_online_hubs(
""",
        """                let (search_result, selected_hub, selected_cached_winners) =
                    searcher.search_with_vertex_hint_ivf_online_hubs(
""",
        "bind per-query cached winner ranking",
    )

    s = once(
        s,
        """                        &online_hubs,
                        &cache_ids,
                    )?;
""",
        """                        &online_hubs,
                        &cache_ids,
                        if winner_snapshot_cadence > 0 {
                            online_hub_capacity.min(10)
                        } else {
                            0
                        },
                    )?;
""",
        "enable cached winner ranking only for snapshot arm",
    )

    s = once(
        s,
        """                if online_hub_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    let bucket = online_hubs.entry(selected_hub).or_default();
""",
        """                // In snapshot mode the hub is populated entirely from
                // query-relevant SUBSETS OF THE EXISTING CACHE. Do not also
                // accumulate per-hub winners independently.
                if winner_snapshot_cadence == 0
                    && online_hub_capacity > 0 && base_count > 0
                {
                    let winner = search_result.results[0].vertex_id;
                    let bucket = online_hubs.entry(selected_hub).or_default();
""",
        "disable redundant per-hub learner in snapshot mode",
    )

    old = """                // Account for the real cache lookup and both online update
                // paths as part of request latency. This includes bookkeeping
                // but deliberately excludes physical page writes not yet made.
"""
    new = r'''                // The cache scan for this request already ranked its
                // preexisting winners. Insert this request's winner first so
                // each persisted page snapshot includes the new evidence.
                if winner_snapshot_cadence > 0
                    && base_count > 0
                    && (qi + 1) % winner_snapshot_cadence == 0
                {
                    let winner = search_result.results[0].vertex_id;
                    let mut subset = Vec::<u32>::with_capacity(online_hub_capacity.min(10));
                    // Preserve the newest successful winner, then the
                    // already-persisted candidates at this hub. Use only the
                    // remaining slots for relevant winners from the existing
                    // value-ID cache. Thus each page rewrite keeps evidence
                    // already paid for, learns the current success, and
                    // packs all ten available slots.
                    subset.push(winner);
                    if let Some(previous) = online_hubs.get(&selected_hub) {
                        for &id in previous {
                            if subset.len() >= online_hub_capacity.min(10) { break; }
                            if !subset.contains(&id) { subset.push(id); }
                        }
                    }
                    for &id in &selected_cached_winners {
                        if subset.len() >= online_hub_capacity.min(10) { break; }
                        if !subset.contains(&id) { subset.push(id); }
                    }

                    let changed = online_hubs.get(&selected_hub)
                        .is_none_or(|old| *old != subset);
                    if changed {
                        if online_hubs.contains_key(&selected_hub) {
                            snapshot_overwrites += 1;
                        }
                        snapshot_total_ids += subset.len();
                        online_hubs.insert(selected_hub, subset);
                        snapshot_unique_pages.insert(selected_hub);
                        snapshot_writes += 1;
                        if qi < online_hub_warmup {
                            snapshot_written_warm += 1;
                        } else {
                            snapshot_written_eval += 1;
                        }
                    } else {
                        snapshot_nochange += 1;
                    }
                }

                // Charge both real PQ lookup and RAM update bookkeeping to
                // request latency. Physical aligned page writes are counted
                // above but are not yet performed in this experiment.
'''
    s = once(s, old, new, "snapshot update from cached winners")

    mark = """            eprintln!(
                "ONLINE_VALUE_STATS L={} capacity={} occupancy={} inserts={} duplicate_skips={} evictions={}",
"""
    replacement = r'''            eprintln!(
                "ONLINE_SNAPSHOT_STATS L={} cadence={} write_ops={} unique_hubs={} overwrites={} unchanged={} winners_written={} warm_writes={} eval_writes={} active_pages={}",
                l,
                winner_snapshot_cadence,
                snapshot_writes,
                snapshot_unique_pages.len(),
                snapshot_overwrites,
                snapshot_nochange,
                snapshot_total_ids,
                snapshot_written_warm,
                snapshot_written_eval,
                online_hubs.len(),
            );
            eprintln!(
                "ONLINE_VALUE_STATS L={} capacity={} occupancy={} inserts={} duplicate_skips={} evictions={}",
'''
    s = once(s, mark, replacement, "snapshot writes diagnostic")
    path.write_text(s)


def main():
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
    print("patched online controller with cache-derived hub-page snapshots")


if __name__ == "__main__":
    main()
