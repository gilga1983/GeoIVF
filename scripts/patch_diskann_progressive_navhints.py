#!/usr/bin/env python3
"""Patch optimized NavHints to retain runner-up hints across graph search.

The canonical 16K Hint-IVF routing pass already PQ-scores the selected coarse
cells and their children. This patch keeps the best 16 scored IDs in the
per-query DiskAccessor instead of discarding all but the winner.

The best ID remains the sole initial start. After each *ordinary* DiskANN beam
finishes, graph search offers the retained IDs in PQ-distance order. At most one
unvisited hint is admitted, and only through DiskANN's existing fixed-L
frontier gate. No beam is split, no second routing pass is performed, and a
rejected hint causes no SSD I/O.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_hint_ivf_packed_direct import (
    PINNED,
    expose_pq_batch_lookup,
    patch_benchmark_global,
    patch_benchmark_hint_ivf_fast,
    patch_provider_hint_ivf_fast,
)
from patch_diskann_paper_catapult import patch_provider_medoid
from patch_diskann_start_points import once, patch_provider as patch_provider_starts
from patch_diskann_waypoint_cache import patch_provider_waypoint


def replace_n(s: str, old: str, new: str, n: int, label: str) -> str:
    count = s.count(old)
    if count != n:
        raise RuntimeError(f"{label}: expected {n} matches, found {count}")
    return s.replace(old, new)


def patch_search_hook(root: Path) -> None:
    glue = root / "diskann/src/graph/glue.rs"
    s = glue.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hook = """    /// Emit query-specific runner-up NavHints retained from start routing.
    /// Ordinary accessors use the no-op default.
    fn progressive_hint_distances<F>(
        &mut self,
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    s = once(s, marker, hook + marker, "progressive SearchAccessor hook")
    glue.write_text(s)

    index = root / "diskann/src/graph/index.rs"
    s = index.read_text()
    anchor = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
"""
    replacement = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                // Preserve the native beam. Only after the whole beam has
                // completed do we consider one dormant runner-up hint.
                let mut progressive_inserted = false;
                accessor
                    .progressive_hint_distances(|id, distance| {
                        if progressive_inserted {
                            return;
                        }
                        let queue_accepts = scratch.best.size() < scratch.best.capacity()
                            || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                        if queue_accepts && scratch.visited.insert(id) {
                            scratch.best.insert(Neighbor::new(id, distance));
                            progressive_inserted = true;
                        }
                    })
                    .await?;
"""
    s = once(s, anchor, replacement, "natural-beam progressive injection")
    index.write_text(s)


def patch_provider_progressive(path: Path) -> None:
    s = path.read_text()

    s = replace_n(
        s,
        """    hint_ivf: Option<HintIvfSearch<'a>>,
""",
        """    hint_ivf: Option<HintIvfSearch<'a>>,
    progressive_hints: bool,
""",
        2,
        "strategy/accessor progressive flag",
    )

    accessor_anchor = """    hint_ivf: Option<HintIvfSearch<'a>>,
    progressive_hints: bool,
}

impl<Data, VP> DiskAccessor<'_, Data, VP>
"""
    s = once(
        s,
        accessor_anchor,
        """    hint_ivf: Option<HintIvfSearch<'a>>,
    progressive_hints: bool,
    retained_hint_ranked: Vec<(u32, f32)>,
}

impl<Data, VP> DiskAccessor<'_, Data, VP>
""",
        "retained hint accessor state",
    )

    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            progressive_hints: strategy.progressive_hints,
            retained_hint_ranked: Vec::new(),
        })
""",
        "progressive accessor constructor",
    )

    s = once(
        s,
        """            start_points,
            hint_ivf: None,
        }
""",
        """            start_points,
            hint_ivf: None,
            progressive_hints: false,
        }
""",
        "generic strategy progressive off",
    )

    s = once(
        s,
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
        }
""",
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
            progressive_hints: false,
        }
""",
        "canonical Hint-IVF progressive off",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    strategy = """    fn search_strategy_with_progressive_hint_ivf<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(hint_ivf),
            progressive_hints: true,
        }
    }

"""
    s = once(s, marker, strategy + marker, "progressive strategy constructor")

    old_start = r'''    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if let Some(ivf) = self.hint_ivf {
            const MAX_NPROBE: usize = 64;
            if ivf.nprobe == 0 || ivf.nprobe > MAX_NPROBE || ivf.nprobe > ivf.medoid_ids.len() {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "invalid Hint-IVF nprobe",
                ));
            }

            // Select the exact same top-nprobe coarse cells as a full
            // canonical sort by (distance, cell), but keep only nprobe entries.
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
                let better_than_worst = distance
                    .total_cmp(&worst.0)
                    .then_with(|| cell.cmp(&worst.1))
                    .is_lt();
                if !better_than_worst {
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

            // Preserve the same canonical fine winner, but gather selected
            // bucket IDs into a Vec owned by DiskANN's pooled scratch. This
            // avoids per-query allocation while retaining one vectorized PQ
            // batch instead of nprobe small batches.
            let mut winner: Option<(f32, u32)> = None;
            let mut routing_cmps = ivf.medoid_ids.len();
            let mut children = std::mem::take(&mut self.scratch.hint_ids_scratch);
            children.clear();
            for &(distance, cell, medoid_id) in coarse.iter() {
                let medoid_better = winner.is_none_or(|current| {
                    distance
                        .total_cmp(&current.0)
                        .then_with(|| medoid_id.cmp(&current.1))
                        .is_lt()
                });
                if medoid_better {
                    winner = Some((distance, medoid_id));
                }

                let lo = ivf.offsets[cell] as usize;
                let hi = ivf.offsets[cell + 1] as usize;
                children.extend_from_slice(&ivf.hint_ids[lo..hi]);
            }
            routing_cmps = routing_cmps.checked_add(children.len()).ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF comparison overflow")
            })?;

            let fine_result = self.pq_distances(&children, |distance, id| {
                let better = winner.is_none_or(|current| {
                    distance
                        .total_cmp(&current.0)
                        .then_with(|| id.cmp(&current.1))
                        .is_lt()
                });
                if better {
                    winner = Some((distance, id));
                }
            });
            children.clear();
            self.scratch.hint_ids_scratch = children;
            fine_result?;

            let winner = winner.ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF selected no start")
            })?;

            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
        }

        if let Some(ids) = self.start_points {
            return self.pq_distances(ids, |dist, id| f(id, dist));
        }
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
'''

    new_start = r'''    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if let Some(ivf) = self.hint_ivf {
            const MAX_NPROBE: usize = 64;
            const RETAINED_TOPK: usize = 16;
            if ivf.nprobe == 0 || ivf.nprobe > MAX_NPROBE || ivf.nprobe > ivf.medoid_ids.len() {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "invalid Hint-IVF nprobe",
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

            let progressive = self.progressive_hints;
            let mut winner: Option<(f32, u32)> = None;
            let mut retained = [(f32::INFINITY, u32::MAX); RETAINED_TOPK];
            let mut consider = |distance: f32, id: u32| {
                if winner.is_none_or(|current| {
                    distance
                        .total_cmp(&current.0)
                        .then_with(|| id.cmp(&current.1))
                        .is_lt()
                }) {
                    winner = Some((distance, id));
                }
                if progressive {
                    let worst = retained[RETAINED_TOPK - 1];
                    if distance
                        .total_cmp(&worst.0)
                        .then_with(|| id.cmp(&worst.1))
                        .is_lt()
                    {
                        let mut pos = RETAINED_TOPK - 1;
                        while pos > 0
                            && distance
                                .total_cmp(&retained[pos - 1].0)
                                .then_with(|| id.cmp(&retained[pos - 1].1))
                                .is_lt()
                        {
                            retained[pos] = retained[pos - 1];
                            pos -= 1;
                        }
                        retained[pos] = (distance, id);
                    }
                }
            };

            let mut routing_cmps = ivf.medoid_ids.len();
            let mut children = std::mem::take(&mut self.scratch.hint_ids_scratch);
            children.clear();
            for &(distance, cell, medoid_id) in coarse.iter() {
                consider(distance, medoid_id);
                let lo = ivf.offsets[cell] as usize;
                let hi = ivf.offsets[cell + 1] as usize;
                children.extend_from_slice(&ivf.hint_ids[lo..hi]);
            }
            routing_cmps = routing_cmps.checked_add(children.len()).ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF comparison overflow")
            })?;

            let fine_result = self.pq_distances(&children, |distance, id| {
                consider(distance, id);
            });
            children.clear();
            self.scratch.hint_ids_scratch = children;
            fine_result?;

            let winner = winner.ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "Hint-IVF selected no start")
            })?;

            if progressive {
                self.retained_hint_ranked.clear();
                self.retained_hint_ranked.extend(
                    retained
                        .into_iter()
                        .filter(|(distance, id)| distance.is_finite() && *id != u32::MAX)
                        .map(|(distance, id)| (id, distance)),
                );
            }

            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
        }

        if let Some(ids) = self.start_points {
            return self.pq_distances(ids, |dist, id| f(id, dist));
        }
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    async fn progressive_hint_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if self.progressive_hints {
            for &(id, distance) in &self.retained_hint_ranked {
                f(id, distance);
            }
        }
        Ok(())
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
'''
    s = once(s, old_start, new_start, "retain runner-up Hint-IVF candidates")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    /// Search from the best NavHint while retaining the runner-ups for
    /// opportunistic admission after ordinary DiskANN beam boundaries.
    pub fn search_with_progressive_hint_ivf(
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
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
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
                "invalid progressive Hint-IVF search shape",
            ));
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
        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
        };
        let strategy = self.search_strategy_with_progressive_hint_ivf(&io_tracker, hint_ivf);
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
    s = once(s, marker, public_method + marker, "progressive Hint-IVF public method")
    path.write_text(s)


def patch_benchmark_progressive(path: Path) -> None:
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
    let progressive_hints = std::env::var("DISKANN_PROGRESSIVE_HINTS")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));

    // Load the vector filters
"""
    s = once(s, load_anchor, load_new, "load progressive flag")

    old_dispatch = """                let result = if let Some(index) = hint_ivf.as_ref() {
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
    new_dispatch = """                let result = if let Some(index) = hint_ivf.as_ref() {
                    if progressive_hints {
                        searcher.search_with_progressive_hint_ivf(
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
                        )
                    }
                } else if active_seeds.is_empty() {
"""
    s = once(s, old_dispatch, new_dispatch, "progressive benchmark dispatch")
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

    expose_pq_batch_lookup(root)
    patch_provider_starts(provider)
    patch_provider_medoid(provider)
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf_fast(provider)
    patch_benchmark_hint_ivf_fast(benchmark)

    patch_search_hook(root)
    patch_provider_progressive(provider)
    patch_benchmark_progressive(benchmark)
    print(f"patched DiskANN {PINNED} with progressive retained NavHints")


if __name__ == "__main__":
    main()
