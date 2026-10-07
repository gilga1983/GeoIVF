#!/usr/bin/env python3
"""Allow the online seed-only value cache to run with vertex hints disabled.

Apply after:
  patch_diskann_vertex_navhints.py
  patch_diskann_hub_winners.py (optional)
  patch_diskann_online_value_cache.py

This is an ablation-only composition patch. It adds the same Hint-IVF +
value-cache search path without selecting a vertex-hint variant, then lets the
chronological replay choose either the vertex-enabled or vertex-disabled path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s=path.read_text()
    marker="""    /// Search with the ordinary Hint-IVF + vertex overlay plus one online
"""
    method=r'''    /// Search with Hint-IVF plus one online value-cache seed, with
    /// vertex continuation hints explicitly disabled.
    pub fn search_with_hint_ivf_value_cache(
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
        value_cache_ids: &[u32],
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if value_cache_ids.is_empty() {
            return self.search_with_hint_ivf(
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
                "invalid no-vertex value-cache Hint-IVF search shape",
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
    s=once(s,marker,method+marker,"no-vertex online value-cache method")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s=path.read_text()

    s=once(
        s,
        """        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
            anyhow::bail!("online value-cache replay requires Hint-IVF and vertex hints");
        }
""",
        """        if hint_ivf.is_none() {
            anyhow::bail!("online value-cache replay requires Hint-IVF");
        }
""",
        "allow no-vertex online cache",
    )

    s=once(
        s,
        """            let variant = vertex_hint_variant
                .ok_or_else(|| anyhow::anyhow!("vertex hints required"))?;
""",
        """            let variant = vertex_hint_variant;
""",
        "optional vertex variant in online replay",
    )

    old=r'''                let search_result = if value_cache_capacity > 0 && !cache_ids.is_empty() {
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
'''
    new=r'''                let search_result = if value_cache_capacity > 0 && !cache_ids.is_empty() {
                    if let Some(variant) = variant {
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
                        searcher.search_with_hint_ivf_value_cache(
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
                            &cache_ids,
                        )?
                    }
                } else if let Some(variant) = variant {
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
                } else {
                    searcher.search_with_hint_ivf(
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
                    )?
                };
'''
    s=once(s,old,new,"optional vertex replay dispatch")
    path.write_text(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("diskann",type=Path);args=ap.parse_args()
    root=args.diskann.resolve()
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("patched online value cache for support-2 ablation")


if __name__=="__main__":main()
