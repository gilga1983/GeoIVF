#!/usr/bin/env python3
"""Exploratory NavHints routing-cache locality ablation on pinned DiskANN3.

Run AFTER patch_diskann_hint_ivf_packed_direct.py and
patch_diskann_experience_core.py. Does not modify graph, junction directory,
PQ algorithm, search width, or the optional persistent-hub mechanism.

Policies:
* fifo: original globally unique Recent512 FIFO (default, exact control).
* lru: same global IDs, but refreshing duplicates changes eviction order.
* region_lru: winners are remembered by the selected Hint-IVF coarse region;
  candidate regions are chosen by the ALREADY SCORED PQ coarse representatives.
  Region buckets share one global item budget and evict the oldest timestamp,
  rather than preallocating a fixed number to every region.

User-space HashMap/Vec metadata and timestamps currently increase real RSS;
B*4 is only a payload accounting convention. This patch is a MECHANISM
experiment, not a publication-grade RAM-equality or latency implementation.
"""
import argparse
from pathlib import Path


def replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one source anchor, got {n}")
    return source.replace(old, new, 1)


def patch_provider(path):
    s=path.read_text()
    s=replace_once(s,
        "    value_cache_ids: Option<&'a [u32]>,\n",
        "    value_cache_ids: Option<&'a [u32]>,\n"
        "    region_cache_ids: Option<&'a [Vec<u32>]>,\n"
        "    region_cache_nprobe: usize,\n",
        "HintIvfSearch region-local view")
    count=s.count("            value_cache_ids: None,\n")
    if count<1:
        raise RuntimeError("No default Hint-IVF constructors to update")
    s=s.replace("            value_cache_ids: None,\n",
                "            value_cache_ids: None,\n"
                "            region_cache_ids: None,\n"
                "            region_cache_nprobe: 1,\n")
    s=replace_once(s,
        "            value_cache_ids: Some(value_cache_ids),\n",
        "            value_cache_ids: Some(value_cache_ids),\n"
        "            region_cache_ids,\n"
        "            region_cache_nprobe,\n",
        "region cache view in experience search")

    s=replace_once(s,
        "    experience_cache_top: std::sync::Mutex<Vec<u32>>,\n",
        "    experience_cache_top: std::sync::Mutex<Vec<u32>>,\n"
        "    selected_hint_region: AtomicUsize,\n",
        "IOTracker selected coarse region")
    s=replace_once(s,
        "            experience_cache_top: std::sync::Mutex::new(Vec::new()),\n",
        "            experience_cache_top: std::sync::Mutex::new(Vec::new()),\n"
        "            selected_hint_region: AtomicUsize::new(usize::MAX),\n",
        "IOTracker regional key initialization")

    # This is the nearest coarse centroid in the *already-scored* Hint-IVF
    # result. We don't hash queries or do a second regional PQ selection.
    s=replace_once(s,
        "            // Preserve the same canonical fine winner, but gather selected\n",
        "            self.io_tracker.selected_hint_region.store(\n"
        "                coarse[0].1,\n"
        "                std::sync::atomic::Ordering::Relaxed,\n"
        "            );\n\n"
        "            // Preserve the same canonical fine winner, but gather selected\n",
        "select nearest coarse region without changing best junction")

    # Score region-specific history with the existing PQ query LUT.
    old='''            let mut emitted = 1usize;
            if let Some(cache_ids) = ivf.value_cache_ids {
                if !cache_ids.is_empty() {
'''
    new='''            let mut emitted = 1usize;
            let mut regional_candidates = Vec::<u32>::new();
            let eligible: Option<&[u32]> = if let Some(groups) = ivf.region_cache_ids {
                let count = ivf.region_cache_nprobe.min(coarse.len());
                for &(_, cell, _) in coarse.iter().take(count) {
                    if let Some(candidates) = groups.get(cell) {
                        regional_candidates.extend_from_slice(candidates);
                    }
                }
                // Duplicates from different query regions add no new route.
                regional_candidates.sort_unstable();
                regional_candidates.dedup();
                Some(regional_candidates.as_slice())
            } else {
                ivf.value_cache_ids
            };
            if let Some(cache_ids) = eligible {
                if !cache_ids.is_empty() {
'''
    s=replace_once(s,old,new,"score conditional destinations instead of global cache")

    # Original top candidate injection as control; optionally emit already
    # PQ-scored top 2/4/8/10. A single OnceLock avoids repeated getenv work.
    old='''                    if let Some(&(distance, id)) = top_cache.first() {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
'''
    new='''                    static ROUTE_RECENT_START_CAP: std::sync::OnceLock<usize> =
                        std::sync::OnceLock::new();
                    let max_additional_starts = *ROUTE_RECENT_START_CAP.get_or_init(|| {
                        std::env::var("DISKANN_LOCAL_RECENT_STARTS")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(1)
                            .min(10)
                    });
                    for &(distance, id) in top_cache.iter().take(max_additional_starts) {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
'''
    s=replace_once(s,old,new,"emit selected top-K destination starts")

    s=replace_once(s,
        "        value_cache_ids: &[u32],\n",
        "        value_cache_ids: &[u32],\n"
        "        region_cache_ids: Option<&[Vec<u32>]>,\n"
        "        region_cache_nprobe: usize,\n",
        "extend experience search arguments")
    s=replace_once(s,
        "    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32, Vec<u32>)> {\n",
        "    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32, usize, Vec<u32>)> {\n",
        "experience search return region key")
    s=replace_once(s,
        "        let cache_top = io_tracker\n",
        "        let selected_region = io_tracker\n"
        "            .selected_hint_region\n"
        "            .load(std::sync::atomic::Ordering::Relaxed);\n"
        "        if selected_region == usize::MAX || selected_region >= medoid_ids.len() {\n"
        "            return Err(diskann_error!(ErrorKind::IndexError,\n"
        "                \"selected Hint-IVF coarse region invalid\"));\n"
        "        }\n"
        "        let cache_top = io_tracker\n",
        "extract selected coarse region")
    s=replace_once(s,
        "        Ok((search_result, selected_hub, cache_top))\n",
        "        Ok((search_result, selected_hub, selected_region, cache_top))\n",
        "return region with search result")
    path.write_text(s)
    print("NAVHINTS_LOCAL_HISTORY_PROVIDER_PATCH_PASS",flush=True)


def patch_benchmark(path):
    s=path.read_text()
    anchor='''    let experience_replay = std::env::var("DISKANN_EXPERIENCE_REPLAY")
'''
    config='''    // Diagnostic modes use exactly the same Hint-IVF junction directory.
    // Regions share one total winner-ID capacity, with demand-driven space.
    let local_history_mode = std::env::var("DISKANN_LOCAL_CACHE_MODE")
        .unwrap_or_else(|_| "fifo".to_string());
    if !matches!(local_history_mode.as_str(), "fifo" | "lru" | "region_lru") {
        anyhow::bail!("unsupported NavHints cache mode");
    }
    let local_region_nprobe = std::env::var("DISKANN_LOCAL_REGION_NPROBE")
        .ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(1);
    if !(1..=8).contains(&local_region_nprobe) {
        anyhow::bail!("NavHints regional cache nprobe out of range 1..=8");
    }
'''
    s=replace_once(s,anchor,config+anchor,"local-history configuration")

    s=replace_once(s,
        "            let mut cache_evictions = 0usize;\n",
        """            let mut cache_evictions = 0usize;
            // For regional LRU, maintain winners per coarse cell.
            // The global item budget adapts to observed regional demand.
            let region_mode = local_history_mode == "region_lru";
            let mut history = if region_mode {
                vec![Vec::<(u32, usize)>::new(); index.medoid_ids.len()]
            } else { Vec::new() };
            let mut history_view = if region_mode {
                vec![Vec::<u32>::new(); index.medoid_ids.len()]
            } else { Vec::new() };
            let mut history_size = 0usize;
""",
        "create query-conditioned destination histories")

    s=replace_once(s,
        "                let (search_result, selected_hub, cache_top) =\n",
        "                let (search_result, selected_hub, selected_region, cache_top) =\n",
        "capture nearest coarse routing cell")
    s=replace_once(s,
        """                        &cache_ids,
                    )?;
""",
        """                        if region_mode { &[] } else { &cache_ids },
                        if region_mode { Some(history_view.as_slice()) } else { None },
                        local_region_nprobe,
                    )?;
""",
        "pass regional view to search")

    old='''                // Update the volatile cache after the completed query, so this
                // request can influence only future requests.
                if experience_cache_capacity > 0 {
                    if cache_members.contains(&winner) {
                        cache_skips += 1;
                    } else {
                        if cache_ids.len() < experience_cache_capacity {
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                        } else {
                            let victim = cache_ids[cache_next];
                            cache_members.remove(&victim);
                            cache_ids[cache_next] = winner;
                            cache_members.insert(winner);
                            cache_next = (cache_next + 1) % experience_cache_capacity;
                            cache_evictions += 1;
                        }
                        cache_inserts += 1;
                    }
                }
'''
    new='''                // The measured query has completed. New observations may
                // influence subsequent queries, never the current query.
                if experience_cache_capacity > 0 {
                    if region_mode {
                        if selected_region >= history.len() {
                            anyhow::bail!("regional cache: selected coarse cell out of bounds");
                        }
                        let position = history[selected_region]
                            .iter().position(|&(id, _)| id == winner);
                        if let Some(pos) = position {
                            history[selected_region].remove(pos);
                            history_view[selected_region].remove(pos);
                            history[selected_region].push((winner, qi));
                            history_view[selected_region].push(winner);
                            cache_skips += 1; // resident and refreshed
                        } else {
                            if history_size == experience_cache_capacity {
                                // Exploratory correctness-first implementation:
                                // evict globally least-recent (region, winner)
                                // pair to allocate the next slot by demand.
                                let mut oldest = (usize::MAX, usize::MAX, usize::MAX);
                                for (ri, entries) in history.iter().enumerate() {
                                    for (pos, &(_, stamp)) in entries.iter().enumerate() {
                                        if stamp < oldest.0 {
                                            oldest = (stamp, ri, pos);
                                        }
                                    }
                                }
                                if oldest.1 == usize::MAX {
                                    anyhow::bail!("regional cache full without an eviction candidate");
                                }
                                history[oldest.1].remove(oldest.2);
                                history_view[oldest.1].remove(oldest.2);
                                cache_evictions += 1;
                            } else {
                                history_size += 1;
                            }
                            history[selected_region].push((winner, qi));
                            history_view[selected_region].push(winner);
                            cache_inserts += 1;
                        }
                    } else if local_history_mode == "lru" {
                        if cache_members.contains(&winner) {
                            let pos = cache_ids.iter().position(|&x| x == winner)
                                .ok_or_else(|| anyhow::anyhow!("global LRU membership disagreement"))?;
                            cache_ids.remove(pos);
                            cache_ids.push(winner);
                            cache_skips += 1; // hit refresh
                        } else {
                            if cache_ids.len() >= experience_cache_capacity {
                                let victim = cache_ids.remove(0);
                                cache_members.remove(&victim);
                                cache_evictions += 1;
                            }
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                            cache_inserts += 1;
                        }
                    } else if cache_members.contains(&winner) {
                        // Original unique FIFO: duplicate DOES NOT refresh.
                        cache_skips += 1;
                    } else {
                        if cache_ids.len() < experience_cache_capacity {
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                        } else {
                            let victim = cache_ids[cache_next];
                            cache_members.remove(&victim);
                            cache_ids[cache_next] = winner;
                            cache_members.insert(winner);
                            cache_next = (cache_next + 1) % experience_cache_capacity;
                            cache_evictions += 1;
                        }
                        cache_inserts += 1;
                    }
                }
'''
    s=replace_once(s,old,new,"replace only the post-query destination cache update")

    s=replace_once(s,
        '''            let persisted_hubs = page_hubs.len();
''',
        '''            eprintln!(
                "NAVHINTS_LOCAL_HISTORY L={} mode={} nprobe={} capacity={} stored={} used_regions={} inserted={} resident_refreshes={} evictions={}",
                l, local_history_mode, local_region_nprobe,
                experience_cache_capacity,
                if region_mode { history_size } else { cache_ids.len() },
                if region_mode { history.iter().filter(|x| !x.is_empty()).count() } else { 1usize },
                cache_inserts, cache_skips, cache_evictions
            );
            let persisted_hubs = page_hubs.len();
''',
        "emit locality mode and cache counts")
    path.write_text(s)
    print("NAVHINTS_LOCAL_HISTORY_BENCHMARK_PATCH_PASS",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("diskann",type=Path)
    args=p.parse_args()
    root=args.diskann.resolve()
    provider=root/"diskann-disk/src/search/provider/disk_provider.rs"
    benchmark=root/"diskann-benchmark/src/disk_index/search.rs"
    if not (provider.is_file() and benchmark.is_file()):
        raise RuntimeError("expected pinned DiskANN checkout")
    patch_provider(provider)
    patch_benchmark(benchmark)

if __name__=="__main__":
    main()
