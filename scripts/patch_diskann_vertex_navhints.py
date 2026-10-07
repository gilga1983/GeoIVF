#!/usr/bin/env python3
"""Patch DiskANN for vertex-granular embedded continuation maps.

Each disk node carries a compile-time number of variants x 4 u32 continuation
IDs in associated data. The default is the original five-arm experiment; larger
packed experiments can select a different variant count with --variants.

The ordinary 16K Hint-IVF remains the bootstrap. After each natural beam, the
closest expanded node's selected four-edge overlay is PQ-scored and at most one
unvisited candidate can enter through DiskANN's normal fixed-L frontier gate.
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

DEFAULT_VARIANTS = 5
SLOTS = 4


def patch_graph_data_type(root: Path, variants: int) -> None:
    total = variants * SLOTS
    path = root / "diskann-disk/src/data_model/graph_data_types.rs"
    s = path.read_text()
    anchor = """    type VectorIdType: VectorId;
}

/// An adhoc `GraphDataType` for implementations that only need the `VectorDataType`
"""
    replacement = f"""    type VectorIdType: VectorId;

    /// Optional research navigation hints carried in associated data.
    fn embedded_navigation_hints(_data: &Self::AssociatedDataType) -> &[u32] {{
        &[]
    }}
}}

/// Search-only graph type carrying {variants} x {SLOTS} continuation IDs.
pub struct VertexNavHints<T, I = u32> {{
    data: std::marker::PhantomData<T>,
    id: std::marker::PhantomData<I>,
}}

impl<T, I> GraphDataType for VertexNavHints<T, I>
where
    T: VectorRepr,
    I: VectorId + 'static,
{{
    type VectorDataType = T;
    type AssociatedDataType = [u32; {total}];
    type VectorIdType = I;

    fn embedded_navigation_hints(data: &Self::AssociatedDataType) -> &[u32] {{
        data
    }}
}}

/// An adhoc `GraphDataType` for implementations that only need the `VectorDataType`
"""
    s = once(s, anchor, replacement, "vertex NavHints graph type")
    path.write_text(s)

    mod = root / "diskann-disk/src/data_model/mod.rs"
    ms = mod.read_text()
    ms = once(
        ms,
        """pub use graph_data_types::{AdHoc, GraphDataType};
""",
        """pub use graph_data_types::{AdHoc, GraphDataType, VertexNavHints};
""",
        "export vertex NavHints type",
    )
    mod.write_text(ms)


def patch_search_hook(root: Path) -> None:
    glue = root / "diskann/src/graph/glue.rs"
    s = glue.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hook = """    /// Score the selected continuation overlay carried by the closest expanded node.
    fn vertex_hint_distances<F>(
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
    s = once(s, marker, hook + marker, "vertex hint SearchAccessor hook")
    glue.write_text(s)

    index = root / "diskann/src/graph/index.rs"
    s = index.read_text()
    anchor = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
"""
    replacement = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                let mut vertex_hint_best: Option<(A::Id, f32)> = None;
                accessor
                    .vertex_hint_distances(&scratch.beam_nodes, |id, distance| {
                        if scratch.visited.contains(&id) {
                            return;
                        }
                        let better = vertex_hint_best
                            .is_none_or(|current| distance.total_cmp(&current.1).is_lt());
                        if better {
                            vertex_hint_best = Some((id, distance));
                        }
                    })
                    .await?;
                if let Some((id, distance)) = vertex_hint_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
"""
    s = once(s, anchor, replacement, "vertex-hint natural-beam injection")
    index.write_text(s)


def patch_provider(path: Path, variants: int) -> None:
    s = path.read_text()

    s = once(
        s,
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,

    /// Selected associated-data continuation variant. None disables the overlay.
    vertex_hint_variant: Option<usize>,
}
""",
        "strategy vertex-hint variant",
    )
    s = once(
        s,
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
    vertex_hint_variant: Option<usize>,
    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
}
""",
        "accessor vertex-hint state",
    )
    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            vertex_hint_variant: strategy.vertex_hint_variant,
            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
        })
""",
        "accessor vertex-hint constructor",
    )
    s = once(
        s,
        """            start_points,
            hint_ivf: None,
        }
""",
        """            start_points,
            hint_ivf: None,
            vertex_hint_variant: None,
        }
""",
        "generic vertex-hint off",
    )
    s = once(
        s,
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
        }
""",
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
            vertex_hint_variant: None,
        }
""",
        "canonical Hint-IVF vertex-hint off",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    helper = """    fn search_strategy_with_vertex_hint_ivf<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
        vertex_hint_variant: usize,
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
        }
    }

"""
    s = once(s, marker, helper + marker, "vertex-hint strategy constructor")

    marker = """    async fn num_starting_points(&self) -> ANNResult<usize> {
"""
    method = r'''    fn vertex_hint_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let Some(variant) = self.vertex_hint_variant else {
                return Ok(());
            };
            if expanded.is_empty() {
                return Ok(());
            }

            const VARIANTS: usize = __VARIANTS__;
            const SLOTS: usize = 4;
            if variant >= VARIANTS {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "vertex-hint variant outside payload",
                ));
            }

            let associated = self
                .scratch
                .vertex_provider
                .get_associated_data(&expanded[0])?;
            let all = Data::embedded_navigation_hints(associated);
            if all.len() != VARIANTS * SLOTS {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "vertex-hint associated-data shape mismatch",
                ));
            }
            let lo = variant * SLOTS;
            let map = &all[lo..lo + SLOTS];

            let mut signature = [u32::MAX; SLOTS];
            signature.copy_from_slice(map);
            if self.vertex_hint_last_valid && signature == self.vertex_hint_last {
                return Ok(());
            }
            self.vertex_hint_last = signature;
            self.vertex_hint_last_valid = true;

            let mut ids = [u32::MAX; SLOTS];
            let mut n = 0usize;
            for &id in map {
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
    method = method.replace("__VARIANTS__", str(variants))
    s = once(s, marker, method + marker, "vertex-hint accessor scoring")

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    pub fn search_with_vertex_hint_ivf(
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
            || vertex_hint_variant >= __VARIANTS__
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid vertex-hint Hint-IVF search shape",
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
    public_method = public_method.replace("__VARIANTS__", str(variants))
    s = once(s, marker, public_method + marker, "vertex-hint public search")
    path.write_text(s)


def patch_benchmark(path: Path, variants: int) -> None:
    s = path.read_text()
    s = once(
        s,
        """    data_model::{AdHoc, CachingStrategy},
""",
        """    data_model::{CachingStrategy, VertexNavHints},
""",
        "vertex-hint benchmark import",
    )
    count = s.count("DiskIndexSearcher::<AdHoc<T>, _>")
    if count != 1:
        raise RuntimeError(f"expected one AdHoc searcher, found {count}")
    s = s.replace(
        "DiskIndexSearcher::<AdHoc<T>, _>",
        "DiskIndexSearcher::<VertexNavHints<T>, _>",
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
    let vertex_hint_variant = std::env::var("DISKANN_VERTEX_HINT_VARIANT")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?;
    if vertex_hint_variant.is_some_and(|v| v >= __VARIANTS__) {
        anyhow::bail!("DISKANN_VERTEX_HINT_VARIANT must be in [0,__MAX_VARIANT__]");
    }
    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    replacement = replacement.replace("__VARIANTS__", str(variants)).replace(
        "__MAX_VARIANT__", str(variants - 1)
    )
    s = once(s, anchor, replacement, "load vertex-hint variant")

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
                    if let Some(variant) = vertex_hint_variant {
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
    s = once(s, old_dispatch, new_dispatch, "vertex-hint benchmark dispatch")
    path.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    ap.add_argument("--variants", type=int, default=DEFAULT_VARIANTS)
    args = ap.parse_args()
    if args.variants <= 0:
        raise SystemExit("--variants must be positive")
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

    patch_graph_data_type(root, args.variants)
    patch_search_hook(root)
    patch_provider(provider, args.variants)
    patch_benchmark(benchmark, args.variants)
    print(
        f"patched DiskANN {PINNED} with vertex-granular embedded NavHints "
        f"({args.variants} variants x {SLOTS} slots)"
    )


if __name__ == "__main__":
    main()
