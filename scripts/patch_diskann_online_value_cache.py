#!/usr/bin/env python3
"""Add a fully online seed-only value-ID cache to vertex-NavHint DiskANN.

Apply after patch_diskann_vertex_navhints.py, and optionally after
patch_diskann_hub_winners.py.

The cache stores only unique database IDs. For each query, DiskANN scans the
currently cached IDs with the same resident PQ query LUT already used by
Hint-IVF, emits the best cached ID as one additional graph start, completes
ordinary search, then inserts the returned rank-1 result iff it is not already
cached. Replacement is insertion FIFO.

No query keys, result siblings, page metadata, or precomputed per-query cache
selection are used. This patch also provides a chronological benchmark replay
mode so cache lookup/update cost is included in QueryStatistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


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
    value_cache_ids: Option<&'a [u32]>,
}
"""
    s = once(s, old, new, "value-cache HintIvfSearch field")

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
            value_cache_ids: None,
        };
"""
    n = s.count(init_old)
    if n < 2:
        raise RuntimeError(f"expected at least two ordinary HintIvfSearch constructors, found {n}")
    s = s.replace(init_old, init_new)

    old_emit = """            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
"""
    new_emit = r'''            // Optionally score a tiny online cache of previously successful
            // database IDs with the same PQ LUT already resident for this query.
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
                    routing_cmps = routing_cmps
                        .checked_add(cache_ids.len())
                        .ok_or_else(|| diskann_error!(
                            ErrorKind::IndexError,
                            "value-cache comparison overflow"
                        ))?;

                    if let Some((distance, id)) = best_cache {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
                }
            }

            // The graph search counts each emitted start once. Record every
            // remaining routing PQ comparison without double counting starts.
            self.io_tracker.routing_comparisons.fetch_add(
                routing_cmps.saturating_sub(emitted),
                std::sync::atomic::Ordering::Relaxed,
            );
            f(winner.1, winner.0);
            return Ok(());
'''
    s = once(s, old_emit, new_emit, "value-cache PQ scan")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    method = r'''    /// Search with the ordinary Hint-IVF + vertex overlay plus one online
    /// value-cache seed selected by scanning cached database IDs with resident PQ.
    pub fn search_with_vertex_hint_ivf_value_cache(
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
        value_cache_ids: &[u32],
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if value_cache_ids.is_empty() {
            return self.search_with_vertex_hint_ivf(
                query,
                return_list_size,
                search_list_size,
                beam_width,
                medoid_ids,
                coarse_local_ids,
                coarse_pq_codes,
                offsets,
                hint_ids,
                nprobe,
                vertex_hint_variant,
            );
        }
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
                "invalid value-cache Hint-IVF search shape",
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
                "value-cache ID outside graph range",
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
            value_cache_ids: Some(value_cache_ids),
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
        Ok(search_result)
    }

'''
    s = once(s, marker, method + marker, "public online value-cache search")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()

    anchor = """    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    config = r'''    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }
    let value_cache_replay = std::env::var("DISKANN_VALUE_CACHE_REPLAY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let value_cache_capacity = std::env::var("DISKANN_VALUE_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let value_cache_warmup = std::env::var("DISKANN_VALUE_CACHE_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4000);
    if value_cache_replay {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
            anyhow::bail!("online value-cache replay requires Hint-IVF and vertex hints");
        }
        if value_cache_warmup >= num_queries {
            anyhow::bail!("DISKANN_VALUE_CACHE_WARMUP must leave measured queries");
        }
    }

    // Load the vector filters
'''
    s = once(s, anchor, config, "online value-cache config")

    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''fn slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("online value-cache GT slicing does not support filtered truth");
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
    s = once(s, marker, helper + marker, "online value-cache GT slice helper")

    start = "        let zipped = queries\n"
    end = "        let total_time = start.elapsed();\n"
    a = s.find(start)
    b = s.find(end, a)
    if a < 0 or b < 0:
        raise RuntimeError("parallel benchmark block markers missing")
    parallel = s[a:b]

    sequential = r'''        if value_cache_replay {
            let index = hint_ivf
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Hint-IVF required"))?;
            let variant = vertex_hint_variant
                .ok_or_else(|| anyhow::anyhow!("vertex hints required"))?;

            // Fixed-size insertion-FIFO directory with exact SkipDup semantics.
            // Cache IDs themselves are the only lookup state.
            let mut cache_ids = Vec::<u32>::with_capacity(value_cache_capacity);
            let mut cache_members = HashSet::<u32>::with_capacity(value_cache_capacity);
            let mut cache_next = 0usize;

            for qi in 0..num_queries {
                let q = queries.row(qi);
                let search_result = if value_cache_capacity > 0 && !cache_ids.is_empty() {
                    searcher.search_with_vertex_hint_ivf_value_cache(
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
                        &cache_ids,
                    )?
                } else {
                    searcher.search_with_vertex_hint_ivf(
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
                    )?
                };

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

                // Query q may influence only future requests.
                if value_cache_capacity > 0 && base_count > 0 {
                    let winner = search_result.results[0].vertex_id;
                    if !cache_members.contains(&winner) {
                        if cache_ids.len() < value_cache_capacity {
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                        } else {
                            let victim = cache_ids[cache_next];
                            cache_members.remove(&victim);
                            cache_ids[cache_next] = winner;
                            cache_members.insert(winner);
                            cache_next = (cache_next + 1) % value_cache_capacity;
                        }
                    }
                }
            }
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
    new_result = """        let search_result = if value_cache_replay {
            let start_q = value_cache_warmup;
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
    s = once(s, old_result, new_result, "online value-cache measured suffix")
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
    print("patched DiskANN with fully online seed-only SkipDup value cache")


if __name__ == "__main__":
    main()
