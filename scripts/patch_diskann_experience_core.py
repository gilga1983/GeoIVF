#!/usr/bin/env python3
"""Add the consolidated causal experience controller to DiskANN.

Apply after patch_diskann_hint_ivf_packed_direct.py. This core patch deliberately
does not use vertex-associated data or the retired support-2 continuation path.

Controller:
  * Hint-IVF 16K chooses the normal learned entry hub.
  * A unique FIFO value cache stores recent successful result IDs and the hub
    that produced them. The best cached ID is PQ-scored as a second start.
  * Persisted hub pages hold up to H successful winner IDs. When the selected
    hub page is expanded, those IDs are PQ-scored as middle-of-search shortcuts.
  * After completion, rank-1 is learned causally for the selected hub and cache.

Hub writes are modeled explicitly. Direct per-hub evidence is protected. On a
flush, spare page slots may be packed with the newest same-hub winners currently
resident in the value cache. Later direct evidence replaces filler on rewrite.
Only the persisted page image participates in hub routing; unflushed direct
evidence remains a write-buffer state and cannot shortcut traversal.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_graph(root: Path) -> None:
    p = root / "diskann/src/graph/glue.rs"
    text = p.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hook = """    /// Score persisted online experience attached to the selected 16K hub.
    fn experience_hub_distances<F>(
        &mut self,
        _expanded: &[Self::Id],
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    text = once(text, marker, hook + marker, "experience SearchAccessor hook")
    p.write_text(text)

    p = root / "diskann/src/graph/index.rs"
    text = p.read_text()
    anchor = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
"""
    insert = r'''                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                let mut experience_candidates: Vec<(A::Id, f32)> = Vec::new();
                accessor
                    .experience_hub_distances(&scratch.beam_nodes, |id, distance| {
                        if !scratch.visited.contains(&id) {
                            experience_candidates.push((id, distance));
                        }
                    })
                    .await?;
                experience_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (id, distance) in experience_candidates {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
'''
    text = once(text, anchor, insert, "experience hub frontier insertion")
    p.write_text(text)


def patch_provider(path: Path) -> None:
    s = path.read_text()

    old = """#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
}
"""
    new = """#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
    experience_hub_pages: Option<&'a std::collections::HashMap<u32, Vec<u32>>>,
    value_cache_ids: Option<&'a [u32]>,
}
"""
    s = once(s, old, new, "experience HintIvfSearch fields")

    init_old = """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
        };
"""
    init_new = """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            experience_hub_pages: None,
            value_cache_ids: None,
        };
"""
    n = s.count(init_old)
    if n < 2:
        raise RuntimeError(f"expected at least two HintIvfSearch constructors, found {n}")
    s = s.replace(init_old, init_new)

    s = once(
        s,
        """    routing_comparisons: AtomicUsize,
}
""",
        """    routing_comparisons: AtomicUsize,
    selected_hint_start: AtomicUsize,
    experience_cache_top: std::sync::Mutex<Vec<u32>>,
}
""",
        "experience selected start tracker",
    )
    s = once(
        s,
        """            routing_comparisons: AtomicUsize::new(0),
        }
""",
        """            routing_comparisons: AtomicUsize::new(0),
            selected_hint_start: AtomicUsize::new(usize::MAX),
            experience_cache_top: std::sync::Mutex::new(Vec::new()),
        }
""",
        "experience selected start init",
    )

    emit_old = """            // The graph search counts the emitted start once. Record the other
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

            // The tiny recent-result cache tries to jump near the answer.
            // Score its database IDs with the same already-built PQ query LUT.
            let mut emitted = 1usize;
            if let Some(cache_ids) = ivf.value_cache_ids {
                if !cache_ids.is_empty() {
                    let mut top_cache = Vec::<(f32, u32)>::with_capacity(10);
                    self.pq_distances(cache_ids, |distance, id| {
                        let pos = top_cache
                            .binary_search_by(|cur| {
                                cur.0
                                    .total_cmp(&distance)
                                    .then_with(|| cur.1.cmp(&id))
                            })
                            .unwrap_or_else(|x| x);
                        if pos < 10 {
                            top_cache.insert(pos, (distance, id));
                            if top_cache.len() > 10 {
                                top_cache.pop();
                            }
                        }
                    })?;
                    routing_cmps = routing_cmps
                        .checked_add(cache_ids.len())
                        .ok_or_else(|| diskann_error!(
                            ErrorKind::IndexError,
                            "experience cache comparison overflow"
                        ))?;

                    {
                        let mut stored = self
                            .io_tracker
                            .experience_cache_top
                            .lock()
                            .map_err(|_| diskann_error!(
                                ErrorKind::IndexError,
                                "experience cache-top lock poisoned"
                            ))?;
                        stored.clear();
                        stored.extend(top_cache.iter().map(|x| x.1));
                    }

                    if let Some(&(distance, id)) = top_cache.first() {
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
    s = once(s, emit_old, emit_new, "experience cache start")

    s = once(
        s,
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
    experience_hub_emitted: bool,
}
""",
        "experience accessor state",
    )
    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            experience_hub_emitted: false,
        })
""",
        "experience accessor init",
    )

    marker = """    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
"""
    scorer = r'''    fn experience_hub_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            if self.experience_hub_emitted || expanded.is_empty() {
                return Ok(());
            }
            let Some(ivf) = self.hint_ivf else {
                return Ok(());
            };
            let Some(pages) = ivf.experience_hub_pages else {
                return Ok(());
            };
            let raw = self
                .io_tracker
                .selected_hint_start
                .load(std::sync::atomic::Ordering::Relaxed);
            if raw == usize::MAX {
                return Ok(());
            }
            let hub = raw as u32;
            if !expanded.contains(&hub) {
                return Ok(());
            }
            self.experience_hub_emitted = true;
            let Some(ids) = pages.get(&hub) else {
                return Ok(());
            };
            if ids.is_empty() {
                return Ok(());
            }
            self.io_tracker.routing_comparisons.fetch_add(
                ids.len(),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(ids, |distance, id| f(id, distance))
        })();
        std::future::ready(result)
    }

'''
    s = once(s, marker, scorer + marker, "experience hub scoring")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    method = r'''    /// Consolidated search: learned 16K entry, optional recent-result cache
    /// start, and persisted winner shortcuts on the selected hub page.
    /// Vertex continuation hints stay disabled in this core path.
    pub fn search_with_hint_ivf_experience(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        medoid_ids: &[u32],
        coarse_local_ids: &[u32],
        coarse_pq_codes: &[u8],
        offsets: &[u32],
        hint_ids: &[u32],
        nprobe: usize,
        experience_hub_pages: &std::collections::HashMap<u32, Vec<u32>>,
        value_cache_ids: &[u32],
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32, Vec<u32>)> {
        if medoid_ids.is_empty()
            || coarse_local_ids.len() != medoid_ids.len()
            || coarse_local_ids
                .iter()
                .enumerate()
                .any(|(i, id)| *id as usize != i)
            || offsets.len() != medoid_ids.len() + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
            || nprobe == 0
            || nprobe > medoid_ids.len()
            || nprobe > 64
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid consolidated experience search shape",
            ));
        }
        if search_list_size < return_list_size {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "search list size must be at least return_list_size",
            ));
        }
        let num_points = self.index.provider().num_points;
        if value_cache_ids.iter().any(|id| *id as usize >= num_points) {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "experience cache ID outside graph range",
            ));
        }

        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; return_list_size as usize];
        let mut distances = vec![0f32; return_list_size as usize];
        let mut associated_data =
            vec![Data::AssociatedDataType::default(); return_list_size as usize];
        let mut result_output_buffer = SearchOutput::new(
            &mut indices,
            &mut distances,
            &mut associated_data,
            None,
        )?;

        let timer = Instant::now();
        let io_tracker = IOTracker::default();
        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            experience_hub_pages: Some(experience_hub_pages),
            value_cache_ids: Some(value_cache_ids),
        };
        let strategy = self.search_strategy_with_hint_ivf(&io_tracker, hint_ivf);
        let knn_search = Knn::new(search_list_size as usize, beam_width)
            .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;

        let stats = self.runtime.block_on(self.index.search(
            knn_search,
            &strategy,
            &DefaultContext,
            query,
            &mut result_output_buffer,
        ))?;

        let selected_raw = io_tracker
            .selected_hint_start
            .load(std::sync::atomic::Ordering::Relaxed);
        if selected_raw == usize::MAX {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "experience search did not select a Hint-IVF hub",
            ));
        }
        let selected_hub = selected_raw as u32;

        let routing_comparisons = io_tracker
            .routing_comparisons
            .load(std::sync::atomic::Ordering::Relaxed) as u32;
        query_stats.total_comparisons = stats.cmps.saturating_add(routing_comparisons);
        query_stats.search_hops = stats.hops;
        query_stats.total_execution_time_us = timer.elapsed().as_micros();
        query_stats.io_time_us = IOTracker::time(&io_tracker.io_time_us) as u128;
        query_stats.total_io_operations = io_tracker.io_count() as u32;
        query_stats.total_vertices_loaded = io_tracker.io_count() as u32;
        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.cpu_time_us = query_stats
            .total_execution_time_us
            .saturating_sub(query_stats.io_time_us)
            .saturating_sub(query_stats.query_pq_preprocess_time_us);

        let mut search_result = SearchResult {
            results: Vec::with_capacity(return_list_size as usize),
            stats: SearchResultStats {
                cmps: query_stats.total_comparisons,
                result_count: stats.result_count,
                query_statistics: query_stats,
            },
        };
        for ((vertex_id, distance), data) in
            indices.into_iter().zip(distances).zip(associated_data)
        {
            search_result.results.push(SearchResultItem {
                vertex_id,
                distance,
                data,
            });
        }
        let cache_top = io_tracker
            .experience_cache_top
            .lock()
            .map_err(|_| diskann_error!(
                ErrorKind::IndexError,
                "experience cache-top lock poisoned"
            ))?
            .clone();
        Ok((search_result, selected_hub, cache_top))
    }

'''
    s = once(s, marker, method + marker, "consolidated experience search")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, time::Instant};
""",
        """use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::atomic::AtomicBool,
    time::Instant,
};
""",
        "experience benchmark imports",
    )

    anchor = """    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }

    // Load the vector filters
"""
    config = r'''    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }
    let experience_replay = std::env::var("DISKANN_EXPERIENCE_REPLAY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let experience_cache_capacity = std::env::var("DISKANN_EXPERIENCE_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let experience_hub_capacity = std::env::var("DISKANN_EXPERIENCE_HUB_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    let experience_flush_threshold = std::env::var("DISKANN_EXPERIENCE_FLUSH_THRESHOLD")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let experience_fill_from_cache = std::env::var("DISKANN_EXPERIENCE_FILL_FROM_CACHE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let experience_warmup = std::env::var("DISKANN_EXPERIENCE_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4000);
    if experience_replay {
        if hint_ivf.is_none() {
            anyhow::bail!("experience replay requires Hint-IVF");
        }
        if experience_warmup >= num_queries {
            anyhow::bail!("experience warmup must leave measured queries");
        }
        if experience_hub_capacity > 0 && experience_flush_threshold == 0 {
            anyhow::bail!("experience flush threshold must be positive");
        }
    }

    // Load the vector filters
'''
    s = once(s, anchor, config, "experience benchmark config")

    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''fn slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("experience GT slicing does not support filtered truth");
    }
    let ids = ctx
        .gt_ids
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GT IDs missing"))?;
    let off = start
        .checked_mul(ctx.gt_dim)
        .ok_or_else(|| anyhow::anyhow!("GT slice overflow"))?;
    Ok(GroundTruthContext {
        gt_ids: Some(ids[off..].to_vec()),
        gt_ids_variable_length: None,
        gt_dists: ctx.gt_dists.as_ref().map(|x| x[off..].to_vec()),
        gt_dim: ctx.gt_dim,
        recall_at: ctx.recall_at,
    })
}

'''
    s = once(s, marker, helper + marker, "experience GT slice helper")

    start = "        let zipped = queries\n"
    end = "        let total_time = start.elapsed();\n"
    a = s.find(start)
    b = s.find(end, a)
    if a < 0 or b < 0:
        raise RuntimeError("parallel benchmark block markers missing")
    parallel = s[a:b]

    sequential = r'''        if experience_replay {
            let index = hint_ivf
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Hint-IVF required"))?;

            // Volatile recent-result cache. IDs are the lookup structure; hub
            // tags and insertion stamps are used only to pack page rewrites.
            let mut cache_ids = Vec::<u32>::with_capacity(experience_cache_capacity);
            let mut cache_members = HashSet::<u32>::with_capacity(experience_cache_capacity);
            let mut cache_next = 0usize;
            let mut cache_inserts = 0usize;
            let mut cache_skips = 0usize;
            let mut cache_evictions = 0usize;

            // Direct evidence is the durable logical truth for each hub.
            // page_hubs is the currently persisted page image visible to search.
            let mut direct_hubs = HashMap::<u32, Vec<u32>>::new();
            let mut page_hubs = HashMap::<u32, Vec<u32>>::new();
            let mut pending = HashMap::<u32, usize>::new();

            let mut direct_learned = 0usize;
            let mut direct_duplicates = 0usize;
            let mut direct_full = 0usize;
            let mut writes = 0usize;
            let mut eval_writes = 0usize;
            let mut write_slots = 0usize;
            let mut write_direct_slots = 0usize;
            let mut write_filler_slots = 0usize;
            let mut eval_write_slots = 0usize;
            let mut eval_write_filler_slots = 0usize;

            for qi in 0..num_queries {
                let q = queries.row(qi);
                let (search_result, selected_hub, cache_top) =
                    searcher.search_with_hint_ivf_experience(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        &index.medoid_ids,
                        &index.coarse_local_ids,
                        &index.coarse_pq_codes,
                        &index.offsets,
                        &index.hint_ids,
                        hint_ivf_nprobe,
                        &page_hubs,
                        &cache_ids,
                    )?;

                let base_count = (search_result.stats.result_count as usize)
                    .min(search_params.recall_at as usize)
                    .min(search_result.results.len());
                let base = qi * search_params.recall_at as usize;
                let id_chunk =
                    &mut result_ids[base..base + search_params.recall_at as usize];
                let dist_chunk =
                    &mut result_dists[base..base + search_params.recall_at as usize];
                id_chunk.fill(0);
                dist_chunk.fill(0.0);
                result_counts[qi] = base_count as u32;
                for (i, item) in search_result.results.iter().take(base_count).enumerate() {
                    id_chunk[i] = item.vertex_id;
                    dist_chunk[i] = item.distance;
                }
                statistics_vec[qi] = search_result.stats.query_statistics;

                if base_count == 0 {
                    continue;
                }
                let winner = search_result.results[0].vertex_id;

                // Update the volatile cache after the completed query, so this
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

                if experience_hub_capacity == 0 {
                    continue;
                }

                let bucket = direct_hubs.entry(selected_hub).or_default();
                let mut added = false;
                if bucket.contains(&winner) {
                    direct_duplicates += 1;
                } else if bucket.len() >= experience_hub_capacity {
                    direct_full += 1;
                } else {
                    bucket.push(winner);
                    direct_learned += 1;
                    *pending.entry(selected_hub).or_insert(0) += 1;
                    added = true;
                }

                let p = pending.get(&selected_hub).copied().unwrap_or(0);
                let should_flush = added
                    && (p >= experience_flush_threshold
                        || bucket.len() >= experience_hub_capacity);
                if !should_flush {
                    continue;
                }

                let direct_snapshot = bucket.clone();
                let mut page = direct_snapshot.clone();

                if experience_fill_from_cache && page.len() < experience_hub_capacity {
                    for &id in &cache_top {
                        if !page.contains(&id) {
                            page.push(id);
                            if page.len() >= experience_hub_capacity {
                                break;
                            }
                        }
                    }
                }

                let direct_slots = direct_snapshot.len().min(page.len());
                let filler_slots = page.len().saturating_sub(direct_slots);
                page_hubs.insert(selected_hub, page.clone());
                pending.insert(selected_hub, 0);

                writes += 1;
                write_slots += page.len();
                write_direct_slots += direct_slots;
                write_filler_slots += filler_slots;
                if qi >= experience_warmup {
                    eval_writes += 1;
                    eval_write_slots += page.len();
                    eval_write_filler_slots += filler_slots;
                }
            }

            let persisted_hubs = page_hubs.len();
            let final_page_slots: usize = page_hubs.values().map(Vec::len).sum();
            let pending_hubs = pending.values().filter(|x| **x > 0).count();
            let pending_entries: usize = pending.values().sum();

            eprintln!(
                "EXPERIENCE_STATS L={} cache_capacity={} hub_capacity={} flush_threshold={} fill={} cache_inserts={} cache_skips={} cache_evictions={} active_direct_hubs={} persisted_hubs={} direct_learned={} direct_duplicates={} direct_full={} writes={} eval_writes={} write_slots={} write_direct_slots={} write_filler_slots={} eval_write_slots={} eval_write_filler_slots={} final_page_slots={} pending_hubs={} pending_entries={}",
                l,
                experience_cache_capacity,
                experience_hub_capacity,
                experience_flush_threshold,
                if experience_fill_from_cache { 1 } else { 0 },
                cache_inserts,
                cache_skips,
                cache_evictions,
                direct_hubs.len(),
                persisted_hubs,
                direct_learned,
                direct_duplicates,
                direct_full,
                writes,
                eval_writes,
                write_slots,
                write_direct_slots,
                write_filler_slots,
                eval_write_slots,
                eval_write_filler_slots,
                final_page_slots,
                pending_hubs,
                pending_entries,
            );
        } else {
''' + parallel + r'''        }
'''
    s = s[:a] + sequential + s[b:]

    old_result = """        let search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;
"""
    new_result = """        let search_result = if experience_replay {
            let start_q = experience_warmup;
            let result_offset = start_q * search_params.recall_at as usize;
            let sliced_gt = slice_ground_truth_context(&gt_context, start_q)?;
            DiskSearchResult::new(
                &statistics_vec[start_q..],
                &result_ids[result_offset..],
                &result_counts[start_q..],
                l,
                total_time.as_secs_f32(),
                num_queries - start_q,
                &sliced_gt,
            )?
        } else {
            DiskSearchResult::new(
                &statistics_vec,
                &result_ids,
                &result_counts,
                l,
                total_time.as_secs_f32(),
                num_queries,
                &gt_context,
            )?
        };
"""
    s = once(s, old_result, new_result, "experience measured suffix")
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    patch_graph(root)
    patch_provider(root / "diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root / "diskann-benchmark/src/disk_index/search.rs")
    print("patched DiskANN with consolidated causal experience controller")


if __name__ == "__main__":
    main()
