#!/usr/bin/env python3
"""Patch pinned DiskANN with one mid-search staged NavHints injection.

Builds on the optimized packed-direct NavHints patch. The generic graph loop
gets two no-op-by-default SearchAccessor hooks:
  * next_staged_hint_hop()
  * staged_hint_distances(hops, callback)

DiskAccessor overrides them only when a second Hint-IVF is configured.
The graph loop aligns the beam boundary exactly to the requested expansion
count, routes one staged hint, and admits it only if DiskANN's fixed-L queue
would accept it. Thus a rejected hint does not poison the visited set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_hint_ivf_packed_direct import (
    PINNED,
    expose_pq_batch_lookup,
    patch_benchmark_global,
    patch_benchmark_hint_ivf_fast,
    patch_provider,
    patch_provider_hint_ivf_fast,
    patch_provider_medoid,
    patch_provider_waypoint,
)
from patch_diskann_start_points import once


def replace_n(s: str, old: str, new: str, n: int, label: str) -> str:
    count = s.count(old)
    if count != n:
        raise RuntimeError(f"{label}: expected {n} matches, found {count}")
    return s.replace(old, new)


def patch_search_hooks(root: Path) -> None:
    glue = root / "diskann/src/graph/glue.rs"
    s = glue.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hooks = """    /// Return the exact expansion count at which a staged navigation hint should
    /// be considered. The default accessor has no staged hint.
    fn next_staged_hint_hop(&self) -> Option<u32> {
        None
    }

    /// Emit staged navigation candidates once their requested expansion count is
    /// reached. Ordinary accessors use the no-op default.
    fn staged_hint_distances<F>(
        &mut self,
        _hops: u32,
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    s = once(s, marker, hooks + marker, "SearchAccessor staged hooks")
    glue.write_text(s)

    index = root / "diskann/src/graph/index.rs"
    s = index.read_text()
    old = """            let mut neighbors = Vec::with_capacity(self.max_degree_with_slack());
            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                scratch.beam_nodes.clear();

                // In this loop we are going to find the beam_width number of nodes that are closest to the query.
                // Each of these nodes will be a frontier node.
                while scratch.beam_nodes.len() < beam_width
                    && let Some(closest_node) = scratch.best.closest_notvisited()
                {
                    search_record.record(closest_node, scratch.hops, scratch.cmps);
                    scratch.beam_nodes.push(*closest_node.id());
                }
"""
    new = """            let mut neighbors = Vec::with_capacity(self.max_degree_with_slack());
            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                // If a staged hint is due, route it before choosing the next beam.
                // Only mark it visited when the same fixed-L frontier gate used by
                // ordinary graph discoveries would admit it.
                if accessor
                    .next_staged_hint_hop()
                    .is_some_and(|hop| scratch.hops >= hop)
                {
                    accessor
                        .staged_hint_distances(scratch.hops, |id, distance| {
                            let queue_accepts = scratch.best.size() < scratch.best.capacity()
                                || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                            if queue_accepts && scratch.visited.insert(id) {
                                scratch.best.insert(Neighbor::new(id, distance));
                                scratch.cmps += 1;
                            }
                        })
                        .await?;
                }

                scratch.beam_nodes.clear();

                // Stop exactly on a staged-hint boundary instead of crossing it
                // with a full beam. After the staged hint is emitted, the accessor
                // reports no next boundary and ordinary beam width resumes.
                let beam_limit = accessor
                    .next_staged_hint_hop()
                    .filter(|hop| *hop > scratch.hops)
                    .map(|hop| beam_width.min((hop - scratch.hops) as usize).max(1))
                    .unwrap_or(beam_width);

                // In this loop we are going to find the beam_width number of nodes that are closest to the query.
                // Each of these nodes will be a frontier node.
                while scratch.beam_nodes.len() < beam_limit
                    && let Some(closest_node) = scratch.best.closest_notvisited()
                {
                    search_record.record(closest_node, scratch.hops, scratch.cmps);
                    scratch.beam_nodes.push(*closest_node.id());
                }
"""
    s = once(s, old, new, "stage-aware graph loop")
    index.write_text(s)


def patch_provider_staged(path: Path) -> None:
    s = path.read_text()

    # Add optional staged directory state to strategy and accessor.
    s = replace_n(
        s,
        """    hint_ivf: Option<HintIvfSearch<'a>>,
""",
        """    hint_ivf: Option<HintIvfSearch<'a>>,
    stage_hint_ivf: Option<HintIvfSearch<'a>>,
    stage_hint_hops: Option<u32>,
""",
        2,
        "strategy/accessor staged fields",
    )

    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            stage_hint_ivf: strategy.stage_hint_ivf,
            stage_hint_hops: strategy.stage_hint_hops,
            stage_hint_emitted: false,
        })
""",
        "accessor staged constructor",
    )

    # stage_hint_emitted belongs only to the accessor, not strategy.
    accessor_anchor = """    stage_hint_ivf: Option<HintIvfSearch<'a>>,
    stage_hint_hops: Option<u32>,
}

impl<Data, VP> DiskAccessor<'_, Data, VP>
"""
    s = once(
        s,
        accessor_anchor,
        """    stage_hint_ivf: Option<HintIvfSearch<'a>>,
    stage_hint_hops: Option<u32>,
    stage_hint_emitted: bool,
}

impl<Data, VP> DiskAccessor<'_, Data, VP>
""",
        "accessor emitted flag",
    )

    # Every existing strategy constructor explicitly disables staging.
    s = s.replace(
        """            hint_ivf: None,
        }
""",
        """            hint_ivf: None,
            stage_hint_ivf: None,
            stage_hint_hops: None,
        }
""",
    )
    s = s.replace(
        """            hint_ivf: Some(hint_ivf),
        }
""",
        """            hint_ivf: Some(hint_ivf),
            stage_hint_ivf: None,
            stage_hint_hops: None,
        }
""",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    helper = """    fn search_strategy_with_staged_hint_ivf<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        start_ivf: HintIvfSearch<'a>,
        stage_ivf: HintIvfSearch<'a>,
        stage_hops: u32,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(start_ivf),
            stage_hint_ivf: Some(stage_ivf),
            stage_hint_hops: Some(stage_hops),
        }
    }

"""
    s = once(s, marker, helper + marker, "staged strategy constructor")

    # Insert staged routing into DiskAccessor's SearchAccessor implementation.
    marker = """    fn expand_beam<Itr, P, F>(
"""
    staged = r'''    fn next_staged_hint_hop(&self) -> Option<u32> {
        if self.stage_hint_emitted {
            None
        } else {
            self.stage_hint_hops
        }
    }

    async fn staged_hint_distances<F>(&mut self, hops: u32, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if self.stage_hint_emitted {
            return Ok(());
        }
        let Some(stage_hops) = self.stage_hint_hops else {
            return Ok(());
        };
        if hops < stage_hops {
            return Ok(());
        }
        let Some(ivf) = self.stage_hint_ivf else {
            return Ok(());
        };

        const MAX_NPROBE: usize = 64;
        if ivf.nprobe == 0 || ivf.nprobe > MAX_NPROBE || ivf.nprobe > ivf.medoid_ids.len() {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid staged Hint-IVF nprobe",
            ));
        }

        let mut coarse_storage =
            [(f32::INFINITY, usize::MAX, u32::MAX); MAX_NPROBE];
        let coarse = &mut coarse_storage[..ivf.nprobe];
        self.pq_distances_packed(
            ivf.coarse_local_ids,
            ivf.coarse_pq_codes,
            |distance, local_id| {
                let cell = local_id as usize;
                let id = ivf.medoid_ids[cell];
                let worst = coarse[ivf.nprobe - 1];
                if distance
                    .total_cmp(&worst.0)
                    .then_with(|| cell.cmp(&worst.1))
                    .is_ge()
                {
                    return;
                }
                let mut pos = ivf.nprobe - 1;
                while pos > 0
                    && distance
                        .total_cmp(&coarse[pos - 1].0)
                        .then_with(|| cell.cmp(&coarse[pos - 1].1))
                        .is_lt()
                {
                    coarse[pos] = coarse[pos - 1];
                    pos -= 1;
                }
                coarse[pos] = (distance, cell, id);
            },
        )?;

        let mut winner: Option<(f32, u32)> = None;
        let mut routing_cmps = ivf.medoid_ids.len();
        let mut children = std::mem::take(&mut self.scratch.hint_ids_scratch);
        children.clear();
        for &(distance, cell, medoid_id) in coarse.iter() {
            if winner.is_none_or(|current| {
                distance
                    .total_cmp(&current.0)
                    .then_with(|| medoid_id.cmp(&current.1))
                    .is_lt()
            }) {
                winner = Some((distance, medoid_id));
            }
            let lo = ivf.offsets[cell] as usize;
            let hi = ivf.offsets[cell + 1] as usize;
            children.extend_from_slice(&ivf.hint_ids[lo..hi]);
        }
        routing_cmps = routing_cmps.checked_add(children.len()).ok_or_else(|| {
            diskann_error!(ErrorKind::IndexError, "staged Hint-IVF comparison overflow")
        })?;

        let fine_result = self.pq_distances(&children, |distance, id| {
            if winner.is_none_or(|current| {
                distance
                    .total_cmp(&current.0)
                    .then_with(|| id.cmp(&current.1))
                    .is_lt()
            }) {
                winner = Some((distance, id));
            }
        });
        children.clear();
        self.scratch.hint_ids_scratch = children;
        fine_result?;

        let winner = winner.ok_or_else(|| {
            diskann_error!(ErrorKind::IndexError, "staged Hint-IVF selected no hint")
        })?;
        self.io_tracker
            .routing_comparisons
            .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
        self.stage_hint_emitted = true;
        f(winner.1, winner.0);
        Ok(())
    }

'''
    s = once(s, marker, staged + marker, "staged DiskAccessor routing")

    # Public staged search method, parallel to search_with_hint_ivf.
    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    /// Search with one start-time and one mid-search ID-only Hint-IVF.
    #[expect(clippy::too_many_arguments)]
    pub fn search_with_staged_hint_ivf(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        start_medoid_ids: &[u32],
        start_coarse_local_ids: &[u32],
        start_coarse_pq_codes: &[u8],
        start_offsets: &[u32],
        start_hint_ids: &[u32],
        start_nprobe: usize,
        stage_medoid_ids: &[u32],
        stage_coarse_local_ids: &[u32],
        stage_coarse_pq_codes: &[u8],
        stage_offsets: &[u32],
        stage_hint_ids: &[u32],
        stage_nprobe: usize,
        stage_hops: u32,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if stage_hops == 0 {
            return Err(diskann_error!(ErrorKind::IndexError, "staged hint hop must be positive"));
        }
        for (medoids, locals, codes, offsets, hints, nprobe) in [
            (
                start_medoid_ids,
                start_coarse_local_ids,
                start_coarse_pq_codes,
                start_offsets,
                start_hint_ids,
                start_nprobe,
            ),
            (
                stage_medoid_ids,
                stage_coarse_local_ids,
                stage_coarse_pq_codes,
                stage_offsets,
                stage_hint_ids,
                stage_nprobe,
            ),
        ] {
            if medoids.is_empty()
                || locals.len() != medoids.len()
                || locals.iter().enumerate().any(|(i, id)| *id as usize != i)
                || offsets.len() != medoids.len() + 1
                || offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(hints.len() as u32)
                || nprobe == 0
                || nprobe > medoids.len()
                || nprobe > 64
            {
                return Err(diskann_error!(ErrorKind::IndexError, "invalid staged Hint-IVF shape"));
            }
            let num_chunks = self.index.provider().pq_data.get_num_chunks();
            if codes.len() != medoids.len() * num_chunks {
                return Err(diskann_error!(ErrorKind::IndexError, "staged coarse PQ size mismatch"));
            }
        }
        if search_list_size < return_list_size {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "search list size must be at least as large as the number of results requested",
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
        let start_ivf = HintIvfSearch {
            medoid_ids: start_medoid_ids,
            coarse_local_ids: start_coarse_local_ids,
            coarse_pq_codes: start_coarse_pq_codes,
            offsets: start_offsets,
            hint_ids: start_hint_ids,
            nprobe: start_nprobe,
        };
        let stage_ivf = HintIvfSearch {
            medoid_ids: stage_medoid_ids,
            coarse_local_ids: stage_coarse_local_ids,
            coarse_pq_codes: stage_coarse_pq_codes,
            offsets: stage_offsets,
            hint_ids: stage_hint_ids,
            nprobe: stage_nprobe,
        };
        let strategy = self.search_strategy_with_staged_hint_ivf(
            &io_tracker,
            start_ivf,
            stage_ivf,
            stage_hops,
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
    s = once(s, marker, public_method + marker, "public staged search")
    path.write_text(s)


def patch_benchmark_staged(path: Path) -> None:
    s = path.read_text()

    load_anchor = """    let hint_ivf_nprobe = std::env::var("DISKANN_HINT_IVF_NPROBE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }

    // Load the vector filters
"""
    load_new = """    let hint_ivf_nprobe = std::env::var("DISKANN_HINT_IVF_NPROBE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }

    let mut stage_hint_ivf = match std::env::var_os("DISKANN_STAGE_HINT_IVF_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let stage_hint_ivf_nprobe = std::env::var("DISKANN_STAGE_HINT_IVF_NPROBE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let stage_hint_hops = std::env::var("DISKANN_STAGE_HINT_HOPS")
        .ok()
        .map(|v| v.parse::<u32>())
        .transpose()?
        .unwrap_or(8);
    if let Some(index) = stage_hint_ivf.as_ref() {
        if hint_ivf.is_none() {
            anyhow::bail!("staged Hint-IVF requires a start Hint-IVF");
        }
        if stage_hint_hops == 0
            || stage_hint_ivf_nprobe == 0
            || stage_hint_ivf_nprobe > index.medoid_ids.len()
            || stage_hint_ivf_nprobe > 64
        {
            anyhow::bail!("invalid staged Hint-IVF parameters");
        }
    }

    // Load the vector filters
"""
    s = once(s, load_anchor, load_new, "load staged Hint-IVF")

    validate_anchor = """    if let Some(index) = hint_ivf.as_mut() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
        index.coarse_local_ids = (0..index.medoid_ids.len() as u32).collect();
        index.coarse_pq_codes = searcher.pack_hint_ivf_coarse_pq(&index.medoid_ids)?;
        eprintln!(
            "Hint-IVF packed coarse PQ cache: {} bytes",
            index.coarse_pq_codes.len()
        );
    }

    logger.log_checkpoint("index_loaded");
"""
    validate_new = """    if let Some(index) = hint_ivf.as_mut() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
        index.coarse_local_ids = (0..index.medoid_ids.len() as u32).collect();
        index.coarse_pq_codes = searcher.pack_hint_ivf_coarse_pq(&index.medoid_ids)?;
        eprintln!(
            "Hint-IVF packed coarse PQ cache: {} bytes",
            index.coarse_pq_codes.len()
        );
    }
    if let Some(index) = stage_hint_ivf.as_mut() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
        index.coarse_local_ids = (0..index.medoid_ids.len() as u32).collect();
        index.coarse_pq_codes = searcher.pack_hint_ivf_coarse_pq(&index.medoid_ids)?;
        eprintln!(
            "staged Hint-IVF packed coarse PQ cache: {} bytes at hop {}",
            index.coarse_pq_codes.len(),
            stage_hint_hops
        );
    }

    logger.log_checkpoint("index_loaded");
"""
    s = once(s, validate_anchor, validate_new, "validate staged Hint-IVF")

    dispatch_anchor = """                let result = if let Some(index) = hint_ivf.as_ref() {
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
                    )
                } else if active_seeds.is_empty() {
"""
    dispatch_new = """                let result = if let (Some(index), Some(stage)) =
                    (hint_ivf.as_ref(), stage_hint_ivf.as_ref())
                {
                    searcher.search_with_staged_hint_ivf(
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
                        &stage.medoid_ids,
                        &stage.coarse_local_ids,
                        &stage.coarse_pq_codes,
                        &stage.offsets,
                        &stage.hint_ids,
                        stage_hint_ivf_nprobe,
                        stage_hint_hops,
                    )
                } else if let Some(index) = hint_ivf.as_ref() {
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
                    )
                } else if active_seeds.is_empty() {
"""
    s = once(s, dispatch_anchor, dispatch_new, "staged benchmark dispatch")
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

    # Apply canonical optimized NavHints first.
    expose_pq_batch_lookup(root)
    patch_provider(provider)
    patch_provider_medoid(provider)
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf_fast(provider)
    patch_benchmark_hint_ivf_fast(benchmark)

    # Then add the single stage boundary and second directory.
    patch_search_hooks(root)
    patch_provider_staged(provider)
    patch_benchmark_staged(benchmark)
    print(f"patched DiskANN {PINNED} with two-stage NavHints")


if __name__ == "__main__":
    main()
