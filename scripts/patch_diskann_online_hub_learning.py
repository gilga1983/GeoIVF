#!/usr/bin/env python3
"""Add causal online learning of successful winners at selected 16K hubs.

Apply after patch_diskann_vertex_navhints.py.

The benchmark maintains a chronological in-memory image of hub-page winner
metadata. A query sees only winners learned by completed earlier queries. The
selected Hint-IVF start hub is captured inside the real search. Once that hub
page is expanded, its currently learned winners are PQ-scored and offered
through DiskANN's ordinary fixed-L frontier. After search completes, the rank-1
result contributes one distinct winner to that selected hub.

This is a logical page-update model. It counts 10-entry batch writes and dirty
residual pages but does not mutate the disk-index file yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_glue(root: Path) -> None:
    p = root / "diskann/src/graph/glue.rs"
    s = p.read_text()
    marker = """    /// Score the selected continuation overlay carried by the closest expanded node.
"""
    hook = """    /// Score winners learned online for the selected 16K start hub once
    /// that hub page is actually expanded.
    fn online_hub_distances<F>(
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
    s = once(s, marker, hook + marker, "online hub SearchAccessor hook")
    p.write_text(s)

    p = root / "diskann/src/graph/index.rs"
    s = p.read_text()
    anchor = """                let mut vertex_hint_best: Option<(A::Id, f32)> = None;
"""
    insert = r'''                let mut online_hub_candidates: Vec<(A::Id, f32)> = Vec::new();
                accessor
                    .online_hub_distances(&scratch.beam_nodes, |id, distance| {
                        if !scratch.visited.contains(&id) {
                            online_hub_candidates.push((id, distance));
                        }
                    })
                    .await?;
                online_hub_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (id, distance) in online_hub_candidates {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }

'''
    s = once(s, anchor, insert + anchor, "online hub frontier insertion")
    p.write_text(s)


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
    online_hub_winners: Option<&'a HashMap<u32, Vec<u32>>>,
}
"""
    s = once(s, old, new, "online hub HintIvfSearch field")

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
            online_hub_winners: None,
        };
"""
    n = s.count(init_old)
    if n < 2:
        raise RuntimeError(f"expected at least two HintIvfSearch constructors, found {n}")
    s = s.replace(init_old, init_new)

    # Capture the exact deployed Hint-IVF winner in per-query tracker state.
    s = once(
        s,
        """    routing_comparisons: AtomicUsize,
}
""",
        """    routing_comparisons: AtomicUsize,
    selected_hint_start: AtomicUsize,
}
""",
        "selected Hint-IVF start tracker",
    )
    s = once(
        s,
        """            routing_comparisons: AtomicUsize::new(0),
        }
""",
        """            routing_comparisons: AtomicUsize::new(0),
            selected_hint_start: AtomicUsize::new(usize::MAX),
        }
""",
        "selected Hint-IVF start tracker init",
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
            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
'''
    s = once(s, emit_old, emit_new, "capture selected Hint-IVF start")

    # Per-query guard prevents re-emitting the same page-resident online list.
    s = once(
        s,
        """    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
}
""",
        """    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
    online_hub_emitted: bool,
}
""",
        "online hub accessor state",
    )
    s = once(
        s,
        """            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
        })
""",
        """            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
            online_hub_emitted: false,
        })
""",
        "online hub accessor init",
    )

    marker = """    fn vertex_hint_distances<F>(
"""
    method = r'''    fn online_hub_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            if self.online_hub_emitted || expanded.is_empty() {
                return Ok(());
            }
            let Some(ivf) = self.hint_ivf else {
                return Ok(());
            };
            let Some(map) = ivf.online_hub_winners else {
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
            self.online_hub_emitted = true;
            let Some(ids) = map.get(&hub) else {
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
    s = once(s, marker, method + marker, "online hub accessor scorer")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    /// Search with causal online winner metadata attached logically to
    /// the selected 16K hub. Returns the exact selected hub ID for post-search learning.
    pub fn search_with_vertex_hint_ivf_online_hubs(
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
        vertex_hint_variant: usize,
        online_hub_winners: &HashMap<u32, Vec<u32>>,
    ) -> ANNResult<(SearchResult<Data::AssociatedDataType>, u32)> {
        if medoid_ids.is_empty()
            || coarse_local_ids.len() != medoid_ids.len()
            || offsets.len() != medoid_ids.len() + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
            || nprobe == 0
            || nprobe > medoid_ids.len()
            || nprobe > 64
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid online-hub Hint-IVF search shape",
            ));
        }
        if search_list_size < return_list_size {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "search list size must be at least return_list_size",
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
            online_hub_winners: Some(online_hub_winners),
        };
        let strategy = self.search_strategy_with_vertex_hint_ivf(
            &io_tracker,
            hint_ivf,
            vertex_hint_variant,
        );
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
                "online-hub search did not select a Hint-IVF start",
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
        Ok((search_result, selected_hub))
    }

'''
    s = once(s, marker, public_method + marker, "public online hub search")
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
        "online hub benchmark imports",
    )

    anchor = """    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    config = r'''    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }
    let online_hub_replay = std::env::var("DISKANN_ONLINE_HUB_REPLAY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let online_hub_capacity = std::env::var("DISKANN_ONLINE_HUB_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    let online_hub_warmup = std::env::var("DISKANN_ONLINE_HUB_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4000);
    let online_hub_flush_batch = std::env::var("DISKANN_ONLINE_HUB_FLUSH_BATCH")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    if online_hub_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
            anyhow::bail!("online hub replay requires Hint-IVF and vertex hints");
        }
        if online_hub_warmup >= num_queries {
            anyhow::bail!("online hub warmup must leave measured queries");
        }
        if online_hub_flush_batch == 0 {
            anyhow::bail!("online hub flush batch must be positive");
        }
    }

    // Load the vector filters
'''
    s = once(s, anchor, config, "online hub benchmark config")

    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''fn slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("online-hub GT slicing does not support filtered truth");
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
    s = once(s, marker, helper + marker, "online hub GT slice helper")

    start = "        let zipped = queries\n"
    end = "        let total_time = start.elapsed();\n"
    a = s.find(start)
    b = s.find(end, a)
    if a < 0 or b < 0:
        raise RuntimeError("parallel benchmark block markers missing")
    parallel = s[a:b]

    sequential = r'''        if online_hub_replay {
            let index = hint_ivf
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Hint-IVF required"))?;
            let variant = vertex_hint_variant
                .ok_or_else(|| anyhow::anyhow!("vertex hints required"))?;

            let mut online_hubs = HashMap::<u32, Vec<u32>>::new();
            let mut pending = HashMap::<u32, usize>::new();
            let mut learned = 0usize;
            let mut duplicates = 0usize;
            let mut full = 0usize;
            let mut batch_writes = 0usize;

            for qi in 0..num_queries {
                let q = queries.row(qi);
                let (search_result, selected_hub) =
                    searcher.search_with_vertex_hint_ivf_online_hubs(
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
                        variant,
                        &online_hubs,
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

                if online_hub_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    let bucket = online_hubs.entry(selected_hub).or_default();
                    if bucket.contains(&winner) {
                        duplicates += 1;
                    } else if bucket.len() >= online_hub_capacity {
                        full += 1;
                    } else {
                        bucket.push(winner);
                        learned += 1;
                        let p = pending.entry(selected_hub).or_insert(0);
                        *p += 1;
                        if *p >= online_hub_flush_batch {
                            batch_writes += 1;
                            *p = 0;
                        }
                    }
                }
            }

            let dirty_pages = pending.values().filter(|x| **x > 0).count();
            let dirty_entries: usize = pending.values().sum();
            eprintln!(
                "ONLINE_HUB_STATS L={} active_hubs={} learned={} duplicates={} full={} batch_writes={} dirty_pages={} dirty_entries={} writes_if_final_flush={}",
                l,
                online_hubs.len(),
                learned,
                duplicates,
                full,
                batch_writes,
                dirty_pages,
                dirty_entries,
                batch_writes + dirty_pages,
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
    new_result = """        let search_result = if online_hub_replay {
            let start_q = online_hub_warmup;
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
    s = once(s, old_result, new_result, "online hub measured suffix")
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
    patch_glue(root)
    patch_provider(provider)
    patch_benchmark(benchmark)
    print("patched DiskANN with causal online hub winner learning")


if __name__ == "__main__":
    main()
