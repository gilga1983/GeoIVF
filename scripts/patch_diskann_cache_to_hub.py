#!/usr/bin/env python3
"""Reuse the existing 512-ID SkipDup cache to periodically seed hub pages.

Apply after patch_diskann_vertex_navhints.py,
patch_diskann_online_hub_learning.py, and
patch_diskann_online_combined.py, in that order.

No second write cache is constructed. Every X completed queries, the selected
16K start hub receives a query-ranked subset of the already-existing 512
successful-winner IDs plus the just-completed rank-1 winner. The PQ lookup
performed for the seed is reused to rank these candidates on flush queries.

The hub-page writes are logical in this first causal qualification: only
future searches can see a new subset, and write counts are reported explicitly.
The next persistence step will execute physical writes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s = path.read_text()
    s = once(s,
        """    value_cache_ids: Option<&'a [u32]>,
}
""",
        """    value_cache_ids: Option<&'a [u32]>,
    capture_cache_write_subset: bool,
}
""", "cache subset capture flag")

    ordinary = """            value_cache_ids: None,
        };
"""
    if s.count(ordinary) < 2:
        raise RuntimeError(f"expected at least two ordinary constructors: {s.count(ordinary)}")
    s = s.replace(ordinary,
        """            value_cache_ids: None,
            capture_cache_write_subset: false,
        };
""")
    s = once(s,
        """            value_cache_ids: Some(value_cache_ids),
        };
""",
        """            value_cache_ids: Some(value_cache_ids),
            capture_cache_write_subset,
        };
""","capture current online cache subset")

    s = once(s,
        """    selected_hint_start: AtomicUsize,
}
""",
        """    selected_hint_start: AtomicUsize,
    cache_write_subset_ids: [AtomicUsize; 10],
}
""","cache subset in IOTracker")

    s = once(s,
        """            selected_hint_start: AtomicUsize::new(usize::MAX),
        }
""",
        """            selected_hint_start: AtomicUsize::new(usize::MAX),
            cache_write_subset_ids: std::array::from_fn(|_| AtomicUsize::new(usize::MAX)),
        }
""","initialize cache subset IOTracker")

    old = r'''                    let mut best_cache: Option<(f32, u32)> = None;
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
'''
    new = r'''                    let mut best_cache: Option<(f32, u32)> = None;
                    let mut write_subset = Vec::<(f32, u32)>::with_capacity(
                        if ivf.capture_cache_write_subset { 10 } else { 0 }
                    );
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
                        if ivf.capture_cache_write_subset {
                            let candidate = (distance, id);
                            let keep = write_subset.len() < 10
                                || distance
                                    .total_cmp(&write_subset.last().unwrap().0)
                                    .then_with(|| id.cmp(&write_subset.last().unwrap().1))
                                    .is_lt();
                            if keep {
                                write_subset.push(candidate);
                                write_subset.sort_unstable_by(|a, b| {
                                    a.0.total_cmp(&b.0)
                                        .then_with(|| a.1.cmp(&b.1))
                                });
                                write_subset.truncate(10);
                            }
                        }
                    })?;
                    if ivf.capture_cache_write_subset {
                        for (i, (_, id)) in write_subset.iter().enumerate() {
                            self.io_tracker.cache_write_subset_ids[i].store(
                                *id as usize, std::sync::atomic::Ordering::Relaxed
                            );
                        }
                    }
'''
    s = once(s, old, new, "score top-10 cache subset with existing PQ scan")

    s = once(s,
        """        online_hub_winners: &HashMap<u32, Vec<u32>>,
        value_cache_ids: &[u32],
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32)> {
""",
        """        online_hub_winners: &HashMap<u32, Vec<u32>>,
        value_cache_ids: &[u32],
        capture_cache_write_subset: bool,
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32, Vec<u32>)> {
""",
        "combined public method: cache write subset input/output")

    old_end = """        Ok((search_result, selected_hub))
    }

"""
    new_end = r'''        let ranked_cached_winners = if capture_cache_write_subset {
            io_tracker.cache_write_subset_ids
                .iter()
                .filter_map(|id| {
                    let raw = id.load(std::sync::atomic::Ordering::Relaxed);
                    if raw == usize::MAX { None } else { Some(raw as u32) }
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok((search_result, selected_hub, ranked_cached_winners))
    }

'''
    s = once(s, old_end, new_end, "return already-scored cache write subset")

    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()
    anchor = """    if combined_value_cache_capacity > 4096 {
        anyhow::bail!("combined cache capacity limit for this screen is 4096");
    }
"""
    replacement = anchor + r'''    let cache_write_every = std::env::var("DISKANN_CACHE_TO_HUB_WRITE_EVERY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let cache_write_winners = std::env::var("DISKANN_CACHE_TO_HUB_WRITE_WINNERS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    if cache_write_every > 0
        && (combined_value_cache_capacity == 0 || online_hub_capacity == 0
            || cache_write_winners == 0 || cache_write_winners > 10
            || cache_write_winners > online_hub_capacity)
    {
        anyhow::bail!("cache-to-hub writes require an online value cache and hub capacity >= subset size");
    }
'''
    s = once(s,anchor,replacement,"configure periodic writes from existing value cache")

    s = once(s,
        """            let mut cache_evictions = 0usize;

            for qi in 0..num_queries {
""",
        """            let mut cache_evictions = 0usize;
            let mut cache_to_hub_writes = 0usize;
            let mut cache_to_hub_unchanged = 0usize;
            let mut cache_to_hub_new_pages = 0usize;
            let mut cache_to_hub_ids_written = 0usize;
            let mut cache_to_hub_full10_writes = 0usize;
            let mut cache_to_hub_short_writes = 0usize;

            for qi in 0..num_queries {
""","initialize periodic cache-to-hub accounting")

    s = once(s,
        """                let (search_result, selected_hub) =
                    searcher.search_with_vertex_hint_ivf_online_hubs(
""",
        """                let should_write_from_cache = cache_write_every > 0
                    && (qi + 1) % cache_write_every == 0;
                let (search_result, selected_hub, ranked_cache_winners) =
                    searcher.search_with_vertex_hint_ivf_online_hubs(
""","choose periodic writer query")

    s = once(s,
        """                        &online_hubs,
                        &cache_ids,
                    )?;
""",
        """                        &online_hubs,
                        &cache_ids,
                        should_write_from_cache,
                    )?;
""","capture cache write candidates from real PQ scan")

    s = once(s,
        """                if online_hub_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    let bucket = online_hubs.entry(selected_hub).or_default();
""",
        """                if cache_write_every == 0 && online_hub_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    let bucket = online_hubs.entry(selected_hub).or_default();
""","disable separate per-hub learner under reuse-cache policy")

    # Update the existing cache exactly as before, then periodically materialize
    # a subset of its successful IDs onto the current central hub.
    anchor = """                // Account for the real cache lookup and both online update
"""
    write = r'''                if should_write_from_cache && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    let mut chosen = Vec::<u32>::with_capacity(cache_write_winners);
                    // Include the completed query first; it could not have
                    // influenced the search that produced it.
                    chosen.push(winner);
                    // Fill this single write to ten distinct successful IDs
                    // whenever the existing value cache has enough entries.
                    for id in ranked_cache_winners {
                        if chosen.len() >= cache_write_winners {
                            break;
                        }
                        if id != winner && !chosen.contains(&id) {
                            chosen.push(id);
                        }
                    }
                    let previous = online_hubs.get(&selected_hub);
                    if previous.is_some_and(|x| *x == chosen) {
                        cache_to_hub_unchanged += 1;
                    } else {
                        if previous.is_none() {
                            cache_to_hub_new_pages += 1;
                        }
                        cache_to_hub_ids_written += chosen.len();
                        if chosen.len() == cache_write_winners {
                            cache_to_hub_full10_writes += 1;
                        } else {
                            cache_to_hub_short_writes += 1;
                        }
                        online_hubs.insert(selected_hub, chosen);
                        cache_to_hub_writes += 1;
                    }
                }

'''
    s = once(s,anchor,write+anchor,"periodically write relevant cached winner subsets")

    anchor = """            let dirty_pages = pending.values().filter(|x| **x > 0).count();
"""
    diag = r'''            eprintln!(
                "CACHE_TO_HUB_WRITES L={} cadence={} slots={} logical_page_writes={} unchanged={} new_pages={} ids_written={} full10_writes={} short_writes={} unique_hub_pages={} writes_per_query={:.6}",
                l,
                cache_write_every,
                cache_write_winners,
                cache_to_hub_writes,
                cache_to_hub_unchanged,
                cache_to_hub_new_pages,
                cache_to_hub_ids_written,
                cache_to_hub_full10_writes,
                cache_to_hub_short_writes,
                online_hubs.len(),
                cache_to_hub_writes as f64 / num_queries as f64,
            );
'''
    s = once(s,anchor,diag+anchor,"log periodic cache-to-hub write amplification")

    path.write_text(s)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("diskann",type=Path)
    args=ap.parse_args()
    root=args.diskann.resolve()
    provider=root/"diskann-disk/src/search/provider/disk_provider.rs"
    benchmark=root/"diskann-benchmark/src/disk_index/search.rs"
    if not provider.is_file() or not benchmark.is_file():
        raise SystemExit("unexpected DiskANN checkout layout")
    patch_provider(provider)
    patch_benchmark(benchmark)
    print("patched periodic hub writes sourced directly from online value-ID cache")


if __name__=="__main__":
    main()
