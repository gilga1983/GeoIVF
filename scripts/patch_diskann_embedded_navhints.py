#!/usr/bin/env python3
"""Patch pinned DiskANN to consume continuation NavHints embedded as associated data.

The disk graph is expected to carry exactly eight u32 continuation IDs per node
in its official associated-data field. The benchmark uses a specialized
GraphDataType whose AssociatedDataType is [u32; 8].

The ordinary 16K optimized Hint-IVF remains the bootstrap start. After each
natural DiskANN beam, the closest just-expanded vertex exposes its embedded
payload. We PQ-score either the first 4 or 8 non-sentinel IDs and offer only
the best unvisited candidate through DiskANN's existing fixed-L frontier gate.

A canonical control arm runs on the exact same enriched disk index but sets the
embedded-hint limit to zero.
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


def patch_graph_data_type(root: Path) -> None:
    path = root / "diskann-disk/src/data_model/graph_data_types.rs"
    s = path.read_text()

    anchor = """    type VectorIdType: VectorId;
}

/// An adhoc 'GraphDataType' for implementations that only need the 'VectorDataType'
"""
    if anchor not in s:
        # Upstream comments use backticks; match exact source spelling.
        anchor = """    type VectorIdType: VectorId;
}

/// An adhoc `GraphDataType` for implementations that only need the `VectorDataType`
"""
    replacement = """    type VectorIdType: VectorId;

    /// Optional research navigation hints carried in associated data.
    /// Ordinary graph data types expose none.
    fn embedded_navigation_hints(_data: &Self::AssociatedDataType) -> &[u32] {
        &[]
    }
}

/// A search-only graph type whose associated data is eight continuation IDs.
pub struct EmbeddedNavHints<T, I = u32> {
    data: std::marker::PhantomData<T>,
    id: std::marker::PhantomData<I>,
}

impl<T, I> GraphDataType for EmbeddedNavHints<T, I>
where
    T: VectorRepr,
    I: VectorId + 'static,
{
    type VectorDataType = T;
    type AssociatedDataType = [u32; 8];
    type VectorIdType = I;

    fn embedded_navigation_hints(data: &Self::AssociatedDataType) -> &[u32] {
        data
    }
}

/// An adhoc `GraphDataType` for implementations that only need the `VectorDataType`
"""
    s = once(s, anchor, replacement, "embedded graph data type")
    path.write_text(s)

    mod = root / "diskann-disk/src/data_model/mod.rs"
    ms = mod.read_text()
    ms = once(
        ms,
        """pub use graph_data_types::{AdHoc, GraphDataType};
""",
        """pub use graph_data_types::{AdHoc, EmbeddedNavHints, GraphDataType};
""",
        "export embedded graph type",
    )
    mod.write_text(ms)


def patch_search_hook(root: Path) -> None:
    glue = root / "diskann/src/graph/glue.rs"
    s = glue.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hook = """    /// Score continuation hints carried by the closest just-expanded vertex.
    /// Ordinary accessors use the no-op default.
    fn embedded_hint_distances<F>(
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
    s = once(s, marker, hook + marker, "embedded SearchAccessor hook")
    glue.write_text(s)

    index = root / "diskann/src/graph/index.rs"
    s = index.read_text()
    anchor = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
"""
    replacement = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                // The node record for every beam member is already resident.
                // Consult only the closest expanded node's tiny associated-data
                // map, then offer at most one candidate through fixed L.
                let mut embedded_best: Option<(A::Id, f32)> = None;
                accessor
                    .embedded_hint_distances(&scratch.beam_nodes, |id, distance| {
                        if scratch.visited.contains(&id) {
                            return;
                        }
                        let better = embedded_best
                            .is_none_or(|current| distance.total_cmp(&current.1).is_lt());
                        if better {
                            embedded_best = Some((id, distance));
                        }
                    })
                    .await?;
                if let Some((id, distance)) = embedded_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
"""
    s = once(s, anchor, replacement, "embedded natural-beam injection")
    index.write_text(s)


def patch_provider_embedded(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,

    /// Number of associated-data continuation IDs to score (0, 4, or 8).
    embedded_hint_limit: usize,
}
""",
        "strategy embedded limit",
    )
    s = once(
        s,
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
    embedded_hint_limit: usize,
    embedded_last: [u32; 8],
    embedded_last_valid: bool,
}
""",
        "accessor embedded state",
    )
    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            embedded_hint_limit: strategy.embedded_hint_limit,
            embedded_last: [u32::MAX; 8],
            embedded_last_valid: false,
        })
""",
        "accessor embedded constructor",
    )
    s = once(
        s,
        """            start_points,
            hint_ivf: None,
        }
""",
        """            start_points,
            hint_ivf: None,
            embedded_hint_limit: 0,
        }
""",
        "generic embedded off",
    )
    s = once(
        s,
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
        }
""",
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
            embedded_hint_limit: 0,
        }
""",
        "canonical Hint-IVF embedded off",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    strategy = """    fn search_strategy_with_embedded_hint_ivf<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
        embedded_hint_limit: usize,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(hint_ivf),
            embedded_hint_limit,
        }
    }

"""
    s = once(s, marker, strategy + marker, "embedded strategy constructor")

    marker = """    async fn num_starting_points(&self) -> ANNResult<usize> {
"""
    method = r'''    fn embedded_hint_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            if self.embedded_hint_limit == 0 || expanded.is_empty() {
                return Ok(());
            }
            if self.embedded_hint_limit > 8 {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "embedded hint limit exceeds payload",
                ));
            }

            let associated = self
                .scratch
                .vertex_provider
                .get_associated_data(&expanded[0])?;
            let hints = Data::embedded_navigation_hints(associated);
            if hints.len() < 8 {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "embedded-hint graph type exposes fewer than eight slots",
                ));
            }

            let mut signature = [u32::MAX; 8];
            signature.copy_from_slice(&hints[..8]);
            if self.embedded_last_valid && signature == self.embedded_last {
                return Ok(());
            }
            self.embedded_last = signature;
            self.embedded_last_valid = true;

            let mut ids = [u32::MAX; 8];
            let mut n = 0usize;
            for &id in hints.iter().take(self.embedded_hint_limit) {
                if id != u32::MAX {
                    ids[n] = id;
                    n += 1;
                }
            }
            if n == 0 {
                return Ok(());
            }

            self.io_tracker.routing_comparisons.fetch_add(
                n,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(&ids[..n], |distance, id| f(id, distance))
        })();
        std::future::ready(result)
    }

'''
    s = once(s, marker, method + marker, "embedded accessor scoring")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    /// Bootstrap with optimized Hint-IVF, then consume continuation
    /// hints already carried in the disk records' associated-data bytes.
    pub fn search_with_embedded_hint_ivf(
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
        embedded_hint_limit: usize,
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
            || !matches!(embedded_hint_limit, 4 | 8)
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid embedded Hint-IVF search shape",
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
        let strategy = self.search_strategy_with_embedded_hint_ivf(
            &io_tracker,
            hint_ivf,
            embedded_hint_limit,
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
    s = once(s, marker, public_method + marker, "embedded public search method")
    path.write_text(s)


def patch_benchmark_embedded(path: Path) -> None:
    s = path.read_text()
    s = once(
        s,
        """    data_model::{AdHoc, CachingStrategy},
""",
        """    data_model::{CachingStrategy, EmbeddedNavHints},
""",
        "embedded benchmark import",
    )
    count = s.count("DiskIndexSearcher::<AdHoc<T>, _>")
    if count != 1:
        raise RuntimeError(f"expected one AdHoc searcher, found {count}")
    s = s.replace(
        "DiskIndexSearcher::<AdHoc<T>, _>",
        "DiskIndexSearcher::<EmbeddedNavHints<T>, _>",
        1,
    )

    anchor = """    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }

    // Load the vector filters
"""
    replacement = """    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
    }
    let embedded_hint_limit = std::env::var("DISKANN_EMBEDDED_HINTS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if !matches!(embedded_hint_limit, 0 | 4 | 8) {
        anyhow::bail!("DISKANN_EMBEDDED_HINTS must be 0, 4, or 8");
    }
    if embedded_hint_limit > 0 && hint_ivf.is_none() {
        anyhow::bail!("embedded hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    s = once(s, anchor, replacement, "load embedded hint limit")

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
                    if embedded_hint_limit > 0 {
                        searcher.search_with_embedded_hint_ivf(
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
                            embedded_hint_limit,
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
    s = once(s, old_dispatch, new_dispatch, "embedded benchmark dispatch")
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

    patch_graph_data_type(root)
    patch_search_hook(root)
    patch_provider_embedded(provider)
    patch_benchmark_embedded(benchmark)
    print(f"patched DiskANN {PINNED} with embedded continuation NavHints")


if __name__ == "__main__":
    main()
