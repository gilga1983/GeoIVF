#!/usr/bin/env python3
"""Add a real online PQ-key semantic result cache to vertex-NavHint DiskANN.

Apply after patch_diskann_vertex_navhints.py.

The cache is scanned inside start_point_distances with DiskANN's already-built PQ
query LUT. The selected prior result[0] is emitted as an extra anchor start. The
remaining successful result IDs are exposed only after that anchor is expanded,
modeling page-resident piggyback metadata with no extra I/O.

Query PQ codes are supplied by DISKANN_SEMANTIC_QUERY_CODES_FILE, a QCPQ0001 file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from patch_diskann_start_points import once

def section_replace(s,begin,end,old,new,label):
    a=s.find(begin); b=s.find(end,a)
    if a<0 or b<0: raise RuntimeError(f"{label}: section missing")
    chunk=s[a:b]; n=chunk.count(old)
    if n!=1: raise RuntimeError(f"{label}: expected 1 found {n}")
    return s[:a]+chunk.replace(old,new,1)+s[b:]

def patch_glue(root:Path):
    p=root/"diskann/src/graph/glue.rs";s=p.read_text()
    marker="""    /// Score the selected continuation overlay carried by the closest expanded node.
"""
    hook="""    /// Score cached sibling result IDs once the semantic-cache anchor is expanded.
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
    s=once(s,marker,hook+marker,"semantic hook");p.write_text(s)
    p=root/"diskann/src/graph/index.rs";s=p.read_text()
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
    s=once(s,anchor,repl,"semantic index insertion");p.write_text(s)

def patch_provider(path:Path):
    s=path.read_text()
    old="""#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
}
"""
    new="""#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
    semantic_query_codes: Option<&'a [u8]>,
    semantic_result_ids: Option<&'a [u32]>,
    semantic_entries: usize,
    semantic_result_width: usize,
}
"""
    s=once(s,old,new,"semantic cache search fields")

    # All ordinary constructors set semantic fields off.
    for begin,end in [
        ("    pub fn search_with_hint_ivf(","    pub fn search_with_vertex_hint_ivf("),
        ("    pub fn search_with_vertex_hint_ivf(","    /// Perform the ordinary graph search from caller-supplied starting vertices."),
    ]:
        oldc="""        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
        };
"""
        newc="""        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            semantic_query_codes: None,
            semantic_result_ids: None,
            semantic_entries: 0,
            semantic_result_width: 0,
        };
"""
        s=section_replace(s,begin,end,oldc,newc,"semantic-off constructor")

    s=once(s,
"""    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
}
""",
"""    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
    semantic_cache_selected: Option<usize>,
    semantic_cache_anchor: Option<u32>,
    semantic_cache_siblings_emitted: bool,
}
""","accessor semantic state")
    s=once(s,
"""            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
        })
""",
"""            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
            semantic_cache_selected: None,
            semantic_cache_anchor: None,
            semantic_cache_siblings_emitted: false,
        })
""","accessor semantic init")

    # Add semantic scorer before vertex hints.
    marker="""    fn vertex_hint_distances<F>(
"""
    scorer=r'''    fn semantic_cache_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            if self.semantic_cache_siblings_emitted {
                return Ok(());
            }
            let Some(ivf) = self.hint_ivf else {
                return Ok(());
            };
            let Some(slot) = self.semantic_cache_selected else {
                return Ok(());
            };
            let Some(anchor) = self.semantic_cache_anchor else {
                return Ok(());
            };
            if !expanded.contains(&anchor) {
                return Ok(());
            }
            let Some(results) = ivf.semantic_result_ids else {
                return Ok(());
            };
            let width = ivf.semantic_result_width;
            if width <= 1 {
                self.semantic_cache_siblings_emitted = true;
                return Ok(());
            }
            let lo = slot
                .checked_mul(width)
                .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "semantic result offset overflow"))?;
            let hi = lo + width;
            if hi > results.len() {
                return Err(diskann_error!(ErrorKind::IndexError, "semantic result slice outside buffer"));
            }
            let siblings = &results[lo + 1..hi];
            self.semantic_cache_siblings_emitted = true;
            self.io_tracker.routing_comparisons.fetch_add(
                siblings.len(),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(siblings, |distance, id| f(id, distance))
        })();
        std::future::ready(result)
    }

'''
    s=once(s,marker,scorer+marker,"semantic scorer")

    # Inject PQ-cache scan after normal Hint-IVF winner selection.
    oldemit="""            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
"""
    newemit=r'''            let mut emitted = 1usize;
            if ivf.semantic_entries > 0 {
                let codes = ivf.semantic_query_codes.ok_or_else(|| {
                    diskann_error!(ErrorKind::IndexError, "semantic query codes missing")
                })?;
                let results = ivf.semantic_result_ids.ok_or_else(|| {
                    diskann_error!(ErrorKind::IndexError, "semantic result IDs missing")
                })?;
                let chunks = self.provider.pq_data.get_num_chunks();
                if codes.len() != ivf.semantic_entries * chunks
                    || ivf.semantic_result_width == 0
                    || results.len() != ivf.semantic_entries * ivf.semantic_result_width
                {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "semantic cache shape mismatch",
                    ));
                }

                let mut local = std::mem::take(&mut self.scratch.hint_ids_scratch);
                local.clear();
                local.extend((0..ivf.semantic_entries).map(|x| x as u32));
                let mut best_cache: Option<(f32, usize)> = None;
                let score_result = self.pq_distances_packed(&local, codes, |distance, local_id| {
                    let slot = local_id as usize;
                    let better = best_cache.is_none_or(|current| {
                        distance
                            .total_cmp(&current.0)
                            .then_with(|| slot.cmp(&current.1))
                            .is_lt()
                    });
                    if better {
                        best_cache = Some((distance, slot));
                    }
                });
                local.clear();
                self.scratch.hint_ids_scratch = local;
                score_result?;
                routing_cmps = routing_cmps
                    .checked_add(ivf.semantic_entries)
                    .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "semantic comparison overflow"))?;

                if let Some((_cache_distance, slot)) = best_cache {
                    let anchor = results[slot * ivf.semantic_result_width];
                    self.semantic_cache_selected = Some(slot);
                    self.semantic_cache_anchor = Some(anchor);
                    if anchor != winner.1 {
                        routing_cmps = routing_cmps
                            .checked_add(1)
                            .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "semantic comparison overflow"))?;
                        self.pq_distances(&[anchor], |distance, id| {
                            emitted += 1;
                            f(id, distance);
                        })?;
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
    s=once(s,oldemit,newemit,"semantic PQ scan and anchor")

    # Add public online-cache method before ordinary starts.
    marker="""    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    method=r'''    pub fn search_with_vertex_hint_ivf_pq_cache(
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
        semantic_query_codes: &[u8],
        semantic_result_ids: &[u32],
        semantic_entries: usize,
        semantic_result_width: usize,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if semantic_entries == 0 || semantic_result_width == 0 {
            return self.search_with_vertex_hint_ivf(
                query, return_list_size, search_list_size, beam_width,
                medoid_ids, coarse_local_ids, coarse_pq_codes, offsets,
                hint_ids, nprobe, vertex_hint_variant,
            );
        }
        if search_list_size < return_list_size {
            return Err(diskann_error!(ErrorKind::IndexError, "search list size below return size"));
        }
        let chunks = self.index.provider().pq_data.get_num_chunks();
        if semantic_query_codes.len() != semantic_entries * chunks
            || semantic_result_ids.len() != semantic_entries * semantic_result_width
        {
            return Err(diskann_error!(ErrorKind::IndexError, "semantic cache public shape mismatch"));
        }

        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; return_list_size as usize];
        let mut distances = vec![0f32; return_list_size as usize];
        let mut associated_data =
            vec![Data::AssociatedDataType::default(); return_list_size as usize];
        let mut result_output_buffer = SearchOutput::new(
            &mut indices, &mut distances, &mut associated_data, None,
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
            semantic_query_codes: Some(semantic_query_codes),
            semantic_result_ids: Some(semantic_result_ids),
            semantic_entries,
            semantic_result_width,
        };
        let strategy = self.search_strategy_with_vertex_hint_ivf(
            &io_tracker, hint_ivf, vertex_hint_variant,
        );
        let knn_search = Knn::new(search_list_size as usize, beam_width)
            .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
        let stats = self.runtime.block_on(self.index.search(
            knn_search, &strategy, &DefaultContext, query, &mut result_output_buffer,
        ))?;
        let routing_comparisons = io_tracker.routing_comparisons
            .load(std::sync::atomic::Ordering::Relaxed) as u32;
        query_stats.total_comparisons = stats.cmps.saturating_add(routing_comparisons);
        query_stats.search_hops = stats.hops;
        query_stats.total_execution_time_us = timer.elapsed().as_micros();
        query_stats.io_time_us = IOTracker::time(&io_tracker.io_time_us) as u128;
        query_stats.total_io_operations = io_tracker.io_count() as u32;
        query_stats.total_vertices_loaded = io_tracker.io_count() as u32;
        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.cpu_time_us = query_stats.total_execution_time_us
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
            search_result.results.push(SearchResultItem { vertex_id, distance, data });
        }
        Ok(search_result)
    }

'''
    s=once(s,marker,method+marker,"public PQ semantic cache search")
    path.write_text(s)

def patch_benchmark(path:Path):
    s=path.read_text()
    s=once(s,
"""use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, time::Instant};
""",
"""use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, time::Instant};
""","benchmark import anchor")
    # config after vertex hints
    anchor="""    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    config=r'''    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }
    let semantic_cache_capacity = std::env::var("DISKANN_SEMANTIC_CACHE_CAPACITY")
        .ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let semantic_cache_warmup = std::env::var("DISKANN_SEMANTIC_CACHE_WARMUP")
        .ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(500);
    let semantic_result_width = std::env::var("DISKANN_SEMANTIC_CACHE_RESULTS")
        .ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(10);
    let semantic_query_codes = if semantic_cache_capacity > 0 {
        let path = std::env::var_os("DISKANN_SEMANTIC_QUERY_CODES_FILE")
            .ok_or_else(|| anyhow::anyhow!("semantic query-code file required"))?;
        let raw = std::fs::read(path)?;
        if raw.len() < 16 || &raw[0..8] != b"QCPQ0001" {
            anyhow::bail!("invalid semantic query-code header");
        }
        let rows = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let chunks = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        if rows != num_queries || chunks == 0 || raw.len() != 16 + rows * chunks {
            anyhow::bail!("semantic query-code shape mismatch");
        }
        Some((chunks, raw[16..].to_vec()))
    } else {
        None
    };
    if semantic_cache_capacity > 0 {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
            anyhow::bail!("PQ semantic cache requires Hint-IVF and vertex hints");
        }
        if semantic_cache_warmup >= num_queries
            || semantic_result_width == 0
            || semantic_result_width > search_params.recall_at as usize
        {
            anyhow::bail!("invalid semantic cache configuration");
        }
    }

    // Load the vector filters
'''
    s=once(s,anchor,config,"PQ semantic config")

    marker="""// Simplified internal structures to reduce parameter count
"""
    helper=r'''fn slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("semantic-cache GT slicing does not support filtered truth");
    }
    let ids = ctx.gt_ids.as_ref().ok_or_else(|| anyhow::anyhow!("GT IDs missing"))?;
    let off = start * ctx.gt_dim;
    Ok(GroundTruthContext {
        gt_ids: Some(ids[off..].to_vec()),
        gt_ids_variable_length: None,
        gt_dists: ctx.gt_dists.as_ref().map(|x| x[off..].to_vec()),
        gt_dim: ctx.gt_dim,
        recall_at: ctx.recall_at,
    })
}

'''
    s=once(s,marker,helper+marker,"GT slice helper")

    # replace parallel block
    start="        let zipped = queries
";end="        let total_time = start.elapsed();
"
    a=s.find(start);b=s.find(end,a)
    if a<0 or b<0: raise RuntimeError("parallel block markers missing")
    parallel=s[a:b]
    sequential=r'''        if semantic_cache_capacity > 0 {
            let index = hint_ivf.as_ref().ok_or_else(|| anyhow::anyhow!("Hint-IVF required"))?;
            let variant = vertex_hint_variant.ok_or_else(|| anyhow::anyhow!("vertex hints required"))?;
            let (semantic_chunks, all_query_codes) = semantic_query_codes
                .as_ref().ok_or_else(|| anyhow::anyhow!("query codes missing"))?;
            let semantic_chunks = *semantic_chunks;

            let mut cache_codes = vec![0u8; semantic_cache_capacity * semantic_chunks];
            let mut cache_results =
                vec![0u32; semantic_cache_capacity * semantic_result_width];
            let mut cache_len = 0usize;
            let mut cache_next = 0usize;

            for qi in 0..num_queries {
                let q = queries.row(qi);
                let vf = &vector_filters[qi];
                let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                    false, vf, search_params.post_processor.as_ref(),
                );

                let use_cache = qi >= semantic_cache_warmup && cache_len > 0;
                let code_slice = if use_cache {
                    &cache_codes[..cache_len * semantic_chunks]
                } else {
                    &[]
                };
                let result_slice = if use_cache {
                    &cache_results[..cache_len * semantic_result_width]
                } else {
                    &[]
                };

                let search_result = if use_cache {
                    searcher.search_with_vertex_hint_ivf_pq_cache(
                        q, search_params.recall_at, l, Some(search_params.beam_width),
                        &index.medoid_ids, &index.coarse_local_ids, &index.coarse_pq_codes,
                        &index.offsets, &index.hint_ids, hint_ivf_nprobe, variant,
                        code_slice, result_slice, cache_len, semantic_result_width,
                    )?
                } else {
                    searcher.search_with_vertex_hint_ivf(
                        q, search_params.recall_at, l, Some(search_params.beam_width),
                        &index.medoid_ids, &index.coarse_local_ids, &index.coarse_pq_codes,
                        &index.offsets, &index.hint_ids, hint_ivf_nprobe, variant,
                    )?
                };

                let base_count = (search_result.stats.result_count as usize)
                    .min(search_params.recall_at as usize)
                    .min(search_result.results.len());
                let base = qi * search_params.recall_at as usize;
                let id_chunk = &mut result_ids[base..base + search_params.recall_at as usize];
                let dist_chunk = &mut result_dists[base..base + search_params.recall_at as usize];
                id_chunk.fill(0); dist_chunk.fill(0.0);
                result_counts[qi] = base_count as u32;
                for (i, item) in search_result.results.iter().take(base_count).enumerate() {
                    id_chunk[i] = item.vertex_id;
                    dist_chunk[i] = item.distance;
                }
                statistics_vec[qi] = search_result.stats.query_statistics;

                if base_count >= semantic_result_width {
                    let slot = if cache_len < semantic_cache_capacity {
                        let s = cache_len;
                        cache_len += 1;
                        s
                    } else {
                        let s = cache_next;
                        cache_next = (cache_next + 1) % semantic_cache_capacity;
                        s
                    };
                    let src_code = &all_query_codes[
                        qi * semantic_chunks..(qi + 1) * semantic_chunks
                    ];
                    cache_codes[slot * semantic_chunks..(slot + 1) * semantic_chunks]
                        .copy_from_slice(src_code);
                    let dst = &mut cache_results[
                        slot * semantic_result_width..(slot + 1) * semantic_result_width
                    ];
                    for (j, item) in search_result.results
                        .iter().take(semantic_result_width).enumerate()
                    {
                        dst[j] = item.vertex_id;
                    }
                }
            }
        } else {
'''+parallel+r'''        }
'''
    s=s[:a]+sequential+s[b:]

    oldres="""        let search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;
"""
    newres="""        let search_result = if semantic_cache_capacity > 0 {
            let start_q = semantic_cache_warmup;
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
                &statistics_vec, &result_ids, &result_counts, l,
                total_time.as_secs_f32(), num_queries, &gt_context,
            )?
        };
"""
    s=once(s,oldres,newres,"measured suffix")
    path.write_text(s)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("diskann",type=Path);args=ap.parse_args()
    root=args.diskann.resolve()
    patch_glue(root)
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("patched DiskANN with online PQ-key page-piggyback semantic cache")
if __name__=="__main__":main()
