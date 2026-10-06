#!/usr/bin/env python3
"""Patch vertex-NavHints DiskANN with query-specific semantic-cache page metadata.

Apply after patch_diskann_vertex_navhints.py.

DISKANN_START_POINTS_FILE rows are interpreted as semantic-cache entries when
Hint-IVF is also enabled:
  entry[0] = cached anchor ID
  entry[1:] = successful sibling result IDs stored on the anchor's page.

At query start DiskANN emits its normal Hint-IVF winner and the cached anchor.
Only after the anchor is actually expanded does the accessor expose and PQ-score
the siblings, modeling page-resident metadata with zero extra I/O.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from patch_diskann_start_points import once

def patch_glue(root:Path)->None:
    p=root/"diskann/src/graph/glue.rs"; s=p.read_text()
    marker="""    /// Score the selected continuation overlay carried by the closest expanded node.
"""
    hook="""    /// Score semantic-cache siblings once their cached anchor has actually been expanded.
    fn semantic_cache_distances<F>(
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
    s=once(s,marker,hook+marker,"semantic cache SearchAccessor hook"); p.write_text(s)

    p=root/"diskann/src/graph/index.rs"; s=p.read_text()
    anchor="""                if let Some((id, distance)) = vertex_hint_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
"""
    repl=anchor+r'''
                let mut semantic_candidates: Vec<(A::Id, f32)> = Vec::new();
                accessor
                    .semantic_cache_distances(&scratch.beam_nodes, |id, distance| {
                        if !scratch.visited.contains(&id) {
                            semantic_candidates.push((id, distance));
                        }
                    })
                    .await?;
                semantic_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (id, distance) in semantic_candidates {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
'''
    s=once(s,anchor,repl,"semantic sibling insertion"); p.write_text(s)

def patch_provider(path:Path)->None:
    s=path.read_text()
    s=once(s,
"""    /// Selected associated-data continuation variant. None disables the overlay.
    vertex_hint_variant: Option<usize>,
}
""",
"""    /// Selected associated-data continuation variant. None disables the overlay.
    vertex_hint_variant: Option<usize>,

    /// Optional semantic-cache page entry: anchor first, then page-resident siblings.
    semantic_cache_entry: Option<&'a [u32]>,
}
""","strategy semantic field")
    s=once(s,
"""    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
}
""",
"""    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
    semantic_cache_entry: Option<&'a [u32]>,
    semantic_cache_siblings_emitted: bool,
}
""","accessor semantic fields")
    s=once(s,
"""            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
        })
""",
"""            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
            semantic_cache_entry: strategy.semantic_cache_entry,
            semantic_cache_siblings_emitted: false,
        })
""","accessor semantic init")
    # Generic constructor.
    s=once(s,
"""            hint_ivf: None,
            vertex_hint_variant: None,
        }
""",
"""            hint_ivf: None,
            vertex_hint_variant: None,
            semantic_cache_entry: None,
        }
""","generic semantic off")
    # Hint-IVF constructor.
    s=once(s,
"""            start_points: None,
            hint_ivf: Some(hint_ivf),
        }
""",
"""            start_points: None,
            hint_ivf: Some(hint_ivf),
            semantic_cache_entry: None,
        }
""","hint ivf semantic off")
    # Vertex Hint-IVF constructor.
    s=once(s,
"""            start_points: None,
            hint_ivf: Some(hint_ivf),
            vertex_hint_variant: Some(vertex_hint_variant),
        }
""",
"""            start_points: None,
            hint_ivf: Some(hint_ivf),
            vertex_hint_variant: Some(vertex_hint_variant),
            semantic_cache_entry: None,
        }
""","vertex semantic off")

    # Add semantic hook implementation immediately before vertex-hint scorer.
    marker="""    fn vertex_hint_distances<F>(
"""
    method=r'''    fn semantic_cache_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let Some(entry) = self.semantic_cache_entry else {
                return Ok(());
            };
            if self.semantic_cache_siblings_emitted || entry.len() <= 1 {
                return Ok(());
            }
            let anchor = entry[0];
            if !expanded.contains(&anchor) {
                return Ok(());
            }
            self.semantic_cache_siblings_emitted = true;
            let siblings = &entry[1..];
            self.io_tracker.routing_comparisons.fetch_add(
                siblings.len(),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(siblings, |distance, id| f(id, distance))
        })();
        std::future::ready(result)
    }

'''
    s=once(s,marker,method+marker,"semantic accessor scorer")

    # Add semantic strategy constructor before ordinary start-points marker.
    marker="""    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    ctor=r'''    fn search_strategy_with_vertex_hint_ivf_semantic_cache<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
        vertex_hint_variant: usize,
        semantic_cache_entry: &'a [u32],
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(hint_ivf),
            vertex_hint_variant: Some(vertex_hint_variant),
            semantic_cache_entry: Some(semantic_cache_entry),
        }
    }

'''
    s=once(s,marker,ctor+marker,"semantic strategy constructor")

    # In Hint-IVF start selection, emit semantic anchor in addition to normal winner.
    old="""            f(winner.1, winner.0);
            return Ok(());
"""
    new=r'''            f(winner.1, winner.0);
            if let Some(entry) = self.semantic_cache_entry {
                if entry.is_empty() {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "semantic cache entry is empty",
                    ));
                }
                let anchor = entry[0];
                if anchor != winner.1 {
                    self.io_tracker.routing_comparisons.fetch_add(
                        1,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    self.pq_distances(&[anchor], |distance, id| f(id, distance))?;
                }
            }
            return Ok(());
'''
    s=once(s,old,new,"emit semantic anchor")

    # Add public method by adapting existing vertex-Hint-IVF search.
    marker="""    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    method=r'''    pub fn search_with_vertex_hint_ivf_semantic_cache(
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
        semantic_cache_entry: &[u32],
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if semantic_cache_entry.is_empty()
            || medoid_ids.is_empty()
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
                "invalid semantic-cache vertex-hint search shape",
            ));
        }
        let num_points = self.index.provider().num_points;
        if semantic_cache_entry.iter().any(|id| *id as usize >= num_points) {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "semantic cache ID outside graph range",
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
        };
        let strategy = self.search_strategy_with_vertex_hint_ivf_semantic_cache(
            &io_tracker,
            hint_ivf,
            vertex_hint_variant,
            semantic_cache_entry,
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
    s=once(s,marker,method+marker,"semantic public search")
    path.write_text(s)

def patch_benchmark(path:Path)->None:
    s=path.read_text()
    old="""                let result = if let Some(index) = hint_ivf.as_ref() {
                    if let Some(variant) = vertex_hint_variant {
                        searcher.search_with_vertex_hint_ivf(
"""
    new="""                let result = if let Some(index) = hint_ivf.as_ref() {
                    if let Some(variant) = vertex_hint_variant {
                        if !seeds.is_empty() {
                            searcher.search_with_vertex_hint_ivf_semantic_cache(
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
                                seeds,
                            )
                        } else {
                            searcher.search_with_vertex_hint_ivf(
"""
    s=once(s,old,new,"semantic dispatch open")
    old2="""                            hint_ivf_nprobe,
                            variant,
                        )
                    } else {
"""
    new2="""                            hint_ivf_nprobe,
                            variant,
                            )
                        }
                    } else {
"""
    s=once(s,old2,new2,"semantic dispatch close")
    path.write_text(s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("diskann",type=Path); args=ap.parse_args()
    root=args.diskann.resolve()
    patch_glue(root)
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("patched vertex NavHints with page-resident semantic-cache siblings")
if __name__=="__main__": main()
