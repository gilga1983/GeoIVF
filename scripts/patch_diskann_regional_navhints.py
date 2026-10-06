#!/usr/bin/env python3
"""Patch optimized NavHints with in-traversal per-region navigation hints.

The initial start remains the existing optimized Hint-IVF NavHint.  A compact
runtime table additionally maps every database vertex to one geometric region
and stores a short learned hint list per region.

After each ordinary DiskANN beam:
  * mark regions represented by the just-expanded vertices as entered;
  * after the configured warmup, select at most ONE newly entered region;
  * PQ-score only that region's small hint list;
  * offer the best unvisited hint through DiskANN's existing fixed-L frontier
    gate.

Thus the graph, beam width, L, SSD records, and full-precision reranking remain
unchanged.  Regional hints add no SSD I/O unless the ordinary frontier accepts
one, and CPU overhead is explicitly visible through routing comparisons and
query CPU time.
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


def patch_search_hook(root: Path) -> None:
    glue = root / "diskann/src/graph/glue.rs"
    s = glue.read_text()
    marker = """    /// A primitive routine used by graph search. This is purposely implemented as a
"""
    hook = """    /// Score hints attached to at most one newly entered graph region.
    /// Ordinary accessors use the no-op default.
    fn regional_hint_distances<F>(
        &mut self,
        _expanded: &[Self::Id],
        _hops: u32,
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    s = once(s, marker, hook + marker, "regional SearchAccessor hook")
    glue.write_text(s)

    index = root / "diskann/src/graph/index.rs"
    s = index.read_text()
    anchor = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
"""
    replacement = """                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                // Regional guidance is considered only at natural beam
                // boundaries. The accessor scores at most one newly entered
                // region per beam; we then admit at most its best unvisited
                // hint through the ordinary fixed-L frontier gate.
                let mut regional_best: Option<(A::Id, f32)> = None;
                accessor
                    .regional_hint_distances(&scratch.beam_nodes, scratch.hops, |id, distance| {
                        if scratch.visited.contains(&id) {
                            return;
                        }
                        let better = regional_best.is_none_or(|current| {
                            distance
                                .total_cmp(&current.1)
                                .then_with(|| id.into_usize().cmp(&current.0.into_usize()))
                                .is_lt()
                        });
                        if better {
                            regional_best = Some((id, distance));
                        }
                    })
                    .await?;
                if let Some((id, distance)) = regional_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }
"""
    s = once(s, anchor, replacement, "regional natural-beam injection")
    index.write_text(s)


def patch_provider_regional(path: Path) -> None:
    s = path.read_text()

    # Borrowed immutable runtime view.
    marker = """pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
"""
    helper = r'''#[derive(Clone, Copy)]
struct RegionalHintSearch<'a> {
    vertex_regions: &'a [u16],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    min_entry_pos: u32,
}

'''
    s = once(s, marker, helper + marker, "regional search view")

    s = once(
        s,
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,

    /// Optional in-traversal regional navigation table.
    regional_hints: Option<RegionalHintSearch<'a>>,
}
""",
        "strategy regional field",
    )

    s = once(
        s,
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        """    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
    regional_hints: Option<RegionalHintSearch<'a>>,
    regional_seen: Vec<bool>,
}
""",
        "accessor regional fields",
    )

    s = once(
        s,
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        """            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
            regional_hints: strategy.regional_hints,
            regional_seen: vec![
                false;
                strategy
                    .regional_hints
                    .map_or(0, |regional| regional.offsets.len().saturating_sub(1))
            ],
        })
""",
        "regional accessor constructor",
    )

    # Every existing non-regional strategy disables the new mechanism.
    s = once(
        s,
        """            start_points,
            hint_ivf: None,
        }
""",
        """            start_points,
            hint_ivf: None,
            regional_hints: None,
        }
""",
        "generic regional off",
    )
    s = once(
        s,
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
        }
""",
        """            start_points: None,
            hint_ivf: Some(hint_ivf),
            regional_hints: None,
        }
""",
        "Hint-IVF regional off",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    strategy = """    fn search_strategy_with_regional_hint_ivf<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
        regional_hints: RegionalHintSearch<'a>,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(hint_ivf),
            regional_hints: Some(regional_hints),
        }
    }

"""
    s = once(s, marker, strategy + marker, "regional strategy constructor")

    # Extend the accessor immediately before num_starting_points(), after the
    # optimized Hint-IVF start selector installed by the base patch.
    marker = """    async fn num_starting_points(&self) -> ANNResult<usize> {
"""
    method = r'''    fn regional_hint_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        hops: u32,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let Some(regional) = self.regional_hints else {
                return Ok(());
            };

            let nregions = regional.offsets.len().saturating_sub(1);
            let mut chosen_region: Option<usize> = None;

            // Mark every region in the beam as entered, but score at most one
            // newly entered region.  The first beam covers positions 0..7 for
            // beam=8, so hops must be strictly greater than min_entry_pos.
            for &id in expanded {
                let pos = id as usize;
                if pos >= regional.vertex_regions.len() {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "regional vertex id outside region table",
                    ));
                }
                let region = regional.vertex_regions[pos] as usize;
                if region >= nregions {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "regional vertex region outside offset table",
                    ));
                }
                if self.regional_seen[region] {
                    continue;
                }
                self.regional_seen[region] = true;
                if chosen_region.is_none() && hops > regional.min_entry_pos {
                    chosen_region = Some(region);
                }
            }

            let Some(region) = chosen_region else {
                return Ok(());
            };
            let lo = regional.offsets[region] as usize;
            let hi = regional.offsets[region + 1] as usize;
            let ids = &regional.hint_ids[lo..hi];
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
    s = once(s, marker, method + marker, "regional accessor scoring")

    # One-time validation entry point.
    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    validate = r'''    pub fn validate_regional_hints(
        &self,
        vertex_regions: &[u16],
        offsets: &[u32],
        hint_ids: &[u32],
    ) -> ANNResult<()> {
        let num_points = self.index.provider().num_points;
        if vertex_regions.len() != num_points {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "regional vertex table length mismatch",
            ));
        }
        if offsets.len() < 2
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(diskann_error!(ErrorKind::IndexError, "invalid regional offsets"));
        }
        let nregions = offsets.len() - 1;
        if vertex_regions.iter().any(|r| (*r as usize) >= nregions) {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "regional vertex label outside region count",
            ));
        }
        if hint_ids.iter().any(|id| (*id as usize) >= num_points) {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "regional hint id outside graph range",
            ));
        }
        Ok(())
    }

'''
    s = once(s, marker, validate + marker, "regional validation method")

    # Clone the optimized Hint-IVF public method shape, adding the regional view.
    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    public_method = r'''    /// Search from the normal optimized NavHint and augment traversal
    /// with equal-budget per-region navigation hints.
    pub fn search_with_regional_hint_ivf(
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
        vertex_regions: &[u16],
        regional_offsets: &[u32],
        regional_hint_ids: &[u32],
        min_entry_pos: u32,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if medoid_ids.is_empty()
            || coarse_local_ids.len() != medoid_ids.len()
            || offsets.len() != medoid_ids.len() + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
            || nprobe == 0
            || nprobe > medoid_ids.len()
            || nprobe > 64
            || regional_offsets.len() < 2
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid regional Hint-IVF search shape",
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
        let regional_hints = RegionalHintSearch {
            vertex_regions,
            offsets: regional_offsets,
            hint_ids: regional_hint_ids,
            min_entry_pos,
        };
        let strategy = self.search_strategy_with_regional_hint_ivf(
            &io_tracker,
            hint_ivf,
            regional_hints,
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
    s = once(s, marker, public_method + marker, "regional public search method")

    path.write_text(s)


def patch_benchmark_regional(path: Path) -> None:
    s = path.read_text()

    # Runtime parser.
    marker = """struct HintIvfIndex {
"""
    parser = r'''struct RegionalHintIndex {
    vertex_regions: Vec<u16>,
    offsets: Vec<u32>,
    hint_ids: Vec<u32>,
    min_entry_pos: u32,
}

impl RegionalHintIndex {
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GIRGN001";
        if raw.len() < 24 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid regional-hint header");
        }
        let nvertices = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let nregions = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let total = u32::from_le_bytes(raw[16..20].try_into()?) as usize;
        let min_entry_pos = u32::from_le_bytes(raw[20..24].try_into()?);
        if nvertices == 0 || nregions == 0 || nregions > u16::MAX as usize {
            anyhow::bail!("invalid regional-hint shape");
        }
        let expected = 24usize
            .checked_add(nvertices.checked_mul(2).ok_or_else(|| anyhow::anyhow!("regional size overflow"))?)
            .and_then(|x| x.checked_add((nregions + 1).checked_mul(4)?))
            .and_then(|x| x.checked_add(total.checked_mul(4)?))
            .ok_or_else(|| anyhow::anyhow!("regional size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "regional-hint byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }
        let mut off = 24usize;
        let mut vertex_regions = Vec::with_capacity(nvertices);
        for _ in 0..nvertices {
            vertex_regions.push(u16::from_le_bytes(raw[off..off + 2].try_into()?));
            off += 2;
        }
        let mut offsets = Vec::with_capacity(nregions + 1);
        for _ in 0..=nregions {
            offsets.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(total as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            anyhow::bail!("invalid regional-hint offsets");
        }
        let mut hint_ids = Vec::with_capacity(total);
        for _ in 0..total {
            hint_ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if off != raw.len() {
            anyhow::bail!("regional-hint parser did not consume all bytes");
        }
        Ok(Self {
            vertex_regions,
            offsets,
            hint_ids,
            min_entry_pos,
        })
    }
}

'''
    s = once(s, marker, parser + marker, "regional runtime parser")

    # Load next to Hint-IVF.
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
    let regional_hints = match std::env::var_os("DISKANN_REGIONAL_HINT_FILE") {
        Some(path) => Some(RegionalHintIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    if regional_hints.is_some() && hint_ivf.is_none() {
        anyhow::bail!("regional hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    s = once(s, anchor, replacement, "load regional hints")

    # Validate once after searcher creation.
    anchor = """    if let Some(index) = hint_ivf.as_mut() {
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
    replacement = """    if let Some(index) = hint_ivf.as_mut() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
        index.coarse_local_ids = (0..index.medoid_ids.len() as u32).collect();
        index.coarse_pq_codes = searcher.pack_hint_ivf_coarse_pq(&index.medoid_ids)?;
        eprintln!(
            "Hint-IVF packed coarse PQ cache: {} bytes",
            index.coarse_pq_codes.len()
        );
    }
    if let Some(regional) = regional_hints.as_ref() {
        searcher.validate_regional_hints(
            &regional.vertex_regions,
            &regional.offsets,
            &regional.hint_ids,
        )?;
        eprintln!(
            "Regional NavHints: {} regions, {} hints, {} vertex labels",
            regional.offsets.len().saturating_sub(1),
            regional.hint_ids.len(),
            regional.vertex_regions.len(),
        );
    }

    logger.log_checkpoint("index_loaded");
"""
    s = once(s, anchor, replacement, "one-time regional validation")

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
                    if let Some(regional) = regional_hints.as_ref() {
                        searcher.search_with_regional_hint_ivf(
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
                            &regional.vertex_regions,
                            &regional.offsets,
                            &regional.hint_ids,
                            regional.min_entry_pos,
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
    s = once(s, old_dispatch, new_dispatch, "regional benchmark dispatch")

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
    patch_provider_regional(provider)
    patch_benchmark_regional(benchmark)
    print(f"patched DiskANN {PINNED} with in-traversal regional NavHints")


if __name__ == "__main__":
    main()
