#!/usr/bin/env python3
"""Patch pinned DiskANN with an optimized ID-only Hint-IVF path.

Compared with patch_diskann_hint_ivf.py, this implementation is deliberately
specialized for the paper's one-start NavHints design:

* Hint routing happens inside DiskANN's SearchAccessor, so the query PQ lookup
  table is prepared once and reused by both Hint-IVF and graph traversal.
* The Hint-IVF file is structurally validated once at load time, and ID ranges
  are validated once after the DiskANN searcher is constructed.
* Per-query routing performs no full vocabulary validation.
* Coarse routing keeps only the best nprobe cells in a fixed stack buffer.
* Selected buckets are PQ-scored directly, without concatenating their IDs.
* With one deployed start, the selector tracks only the best hint rather than
  materializing and sorting the complete fine candidate set.

No full vectors, copied PQ codes, adjacency lists, or auxiliary graph are kept.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patch_diskann_start_points import PINNED, once, patch_provider
from patch_diskann_paper_catapult import patch_provider_medoid
from patch_diskann_waypoint_cache import patch_provider_waypoint
from patch_diskann_global_starts import patch_benchmark_global


def patch_provider_hint_ivf_fast(path: Path) -> None:
    s = path.read_text()

    # A borrowed immutable view over the already-validated ID-only directory.
    marker = """pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
"""
    helper = r'''#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
}

'''
    s = once(s, marker, helper + marker, "hint-IVF search view")

    # Carry the optional dynamic start policy through the already-existing
    # search strategy and accessor. Ordinary DiskANN/custom-start behavior
    # remains None and therefore unchanged.
    s = once(
        s,
        """    /// Optional query-specific graph entry points. None preserves the upstream medoid.
    start_points: Option<&'a [u32]>,
}
""",
        """    /// Optional query-specific graph entry points. None preserves the upstream medoid.
    start_points: Option<&'a [u32]>,

    /// Optional ID-only Hint-IVF selector. Mutually exclusive with start_points.
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        "strategy hint-IVF field",
    )

    s = once(
        s,
        """    flat_scan_filter: Option<IdFilter<'a>>,
    start_points: Option<&'a [u32]>,
}
""",
        """    flat_scan_filter: Option<IdFilter<'a>>,
    start_points: Option<&'a [u32]>,
    hint_ivf: Option<HintIvfSearch<'a>>,
}
""",
        "accessor hint-IVF field",
    )

    s = once(
        s,
        """            flat_scan_filter: None,
            start_points: strategy.start_points,
        })
""",
        """            flat_scan_filter: None,
            start_points: strategy.start_points,
            hint_ivf: strategy.hint_ivf,
        })
""",
        "accessor hint-IVF constructor",
    )

    # Existing generic strategy construction always disables Hint-IVF.
    s = once(
        s,
        """            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points,
        }
""",
        """            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points,
            hint_ivf: None,
        }
""",
        "disable Hint-IVF in generic strategy constructor",
    )

    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    helper = """    fn search_strategy_with_hint_ivf<'a>(
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
        }
    }

"""
    s = once(s, marker, helper + marker, "Hint-IVF strategy constructor")

    # Replace only the starting-point distance routine. The same DiskAccessor
    # remains alive for expand_beam, therefore the PQ query LUT is reused.
    old = """    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if let Some(ids) = self.start_points {
            return self.pq_distances(ids, |dist, id| f(id, dist));
        }
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
"""
    new = r'''    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
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

            // Keep the best nprobe coarse cells in a tiny fixed stack buffer.
            // Distances use the already-prepared PQ table in this accessor.
            let mut best_storage = [(f32::INFINITY, usize::MAX, u32::MAX); MAX_NPROBE];
            let best_cells = &mut best_storage[..ivf.nprobe];
            let mut cell = 0usize;
            self.pq_distances(ivf.medoid_ids, |distance, id| {
                if distance < best_cells[ivf.nprobe - 1].0 {
                    let mut pos = ivf.nprobe - 1;
                    while pos > 0 && distance < best_cells[pos - 1].0 {
                        best_cells[pos] = best_cells[pos - 1];
                        pos -= 1;
                    }
                    best_cells[pos] = (distance, cell, id);
                }
                cell += 1;
            })?;

            // One start is intentional: DiskANN's released L budget remains
            // unchanged. Representatives are themselves valid learned hints.
            let mut winner = (best_cells[0].0, best_cells[0].2);
            let mut routing_cmps = ivf.medoid_ids.len();

            for &(medoid_distance, coarse_cell, medoid_id) in best_cells.iter() {
                if medoid_distance < winner.0 {
                    winner = (medoid_distance, medoid_id);
                }
                let lo = ivf.offsets[coarse_cell] as usize;
                let hi = ivf.offsets[coarse_cell + 1] as usize;
                let bucket = &ivf.hint_ids[lo..hi];
                routing_cmps = routing_cmps.checked_add(bucket.len()).ok_or_else(|| {
                    diskann_error!(ErrorKind::IndexError, "Hint-IVF comparison overflow")
                })?;
                self.pq_distances(bucket, |distance, id| {
                    if distance < winner.0 {
                        winner = (distance, id);
                    }
                })?;
            }

            // The graph search counts the emitted start once. Record the other
            // routing PQ scores separately so QueryStatistics can report true
            // end-to-end distance work without double-counting the winner.
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
    s = once(s, old, new, "integrated Hint-IVF start selection")

    # Add one cheap counter to the existing per-query IO tracker.
    s = once(
        s,
        """struct IOTracker {
    io_time_us: AtomicU64,
    preprocess_time_us: AtomicU64,
    io_count: AtomicUsize,
}
""",
        """struct IOTracker {
    io_time_us: AtomicU64,
    preprocess_time_us: AtomicU64,
    io_count: AtomicUsize,
    routing_comparisons: AtomicUsize,
}
""",
        "routing comparison counter",
    )
    s = once(
        s,
        """            preprocess_time_us: AtomicU64::new(0),
            io_count: AtomicUsize::new(0),
        }
""",
        """            preprocess_time_us: AtomicU64::new(0),
            io_count: AtomicUsize::new(0),
            routing_comparisons: AtomicUsize::new(0),
        }
""",
        "routing comparison counter init",
    )

    # One-time range validation. Structural/uniqueness validation is performed
    # by the benchmark parser when the file is loaded.
    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    methods = r'''    /// Validate Hint-IVF database IDs once before serving queries.
    pub fn validate_hint_ivf_ids(
        &self,
        medoid_ids: &[u32],
        hint_ids: &[u32],
    ) -> ANNResult<()> {
        let num_points = self.index.provider().num_points;
        if medoid_ids
            .iter()
            .chain(hint_ids.iter())
            .any(|id| (*id as usize) >= num_points)
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "Hint-IVF ID outside graph range",
            ));
        }
        Ok(())
    }

    /// Search with one dynamically selected ID-only Hint-IVF start.
    ///
    /// The selector runs inside the same DiskAccessor as graph traversal, so
    /// DiskANN prepares the query PQ lookup table exactly once.
    pub fn search_with_hint_ivf(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        medoid_ids: &[u32],
        offsets: &[u32],
        hint_ids: &[u32],
        nprobe: usize,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if medoid_ids.is_empty()
            || offsets.len() != medoid_ids.len() + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
            || nprobe == 0
            || nprobe > medoid_ids.len()
            || nprobe > 64
        {
            return Err(diskann_error!(ErrorKind::IndexError, "invalid Hint-IVF search shape"));
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
            offsets,
            hint_ids,
            nprobe,
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
    s = once(s, marker, methods + marker, "optimized Hint-IVF public methods")

    path.write_text(s)


def patch_benchmark_hint_ivf_fast(path: Path) -> None:
    s = path.read_text()

    # Load the same on-disk format as the reference implementation.
    old_load = """    let global_start_ids = match std::env::var_os("DISKANN_GLOBAL_START_IDS_FILE") {
        Some(path) => load_global_start_ids(std::path::Path::new(&path))?,
        None => Vec::<u32>::new(),
    };

    // Load the vector filters
"""
    new_load = """    let global_start_ids = match std::env::var_os("DISKANN_GLOBAL_START_IDS_FILE") {
        Some(path) => load_global_start_ids(std::path::Path::new(&path))?,
        None => Vec::<u32>::new(),
    };
    let hint_ivf = match std::env::var_os("DISKANN_HINT_IVF_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let hint_ivf_nprobe = std::env::var("DISKANN_HINT_IVF_NPROBE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() || hint_ivf_nprobe > 64 {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
        if search_params.vector_filters_file.is_some()
            || search_params.post_processor.is_some()
            || search_params.search_mode.is_flat_search
        {
            anyhow::bail!(
                "optimized Hint-IVF currently supports only unfiltered graph search"
            );
        }
    }

    // Load the vector filters
"""
    s = once(s, old_load, new_load, "load optimized Hint-IVF")

    # Perform expensive database-range validation once, not once per query.
    old_searcher = """    let searcher = &DiskIndexSearcher::<AdHoc<T>, _>::new(
        search_params.num_threads,
        if let Some(lim) = search_params.search_io_limit {
            lim
        } else {
            usize::MAX
        },
        &index_reader,
        vertex_provider_factory,
        search_params.distance.into(),
        None,
    )?;

    logger.log_checkpoint("index_loaded");
"""
    new_searcher = """    let searcher = &DiskIndexSearcher::<AdHoc<T>, _>::new(
        search_params.num_threads,
        if let Some(lim) = search_params.search_io_limit {
            lim
        } else {
            usize::MAX
        },
        &index_reader,
        vertex_provider_factory,
        search_params.distance.into(),
        None,
    )?;
    if let Some(index) = hint_ivf.as_ref() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
    }

    logger.log_checkpoint("index_loaded");
"""
    s = once(s, old_searcher, new_searcher, "one-time Hint-IVF ID validation")

    # Dispatch directly to the integrated selector. Global-flat and per-query
    # custom starts remain available as control arms.
    old_call = """                let active_seeds: &[u32] = if !global_start_ids.is_empty() {
                    global_start_ids.as_slice()
                } else {
                    seeds.as_slice()
                };
                let result = if active_seeds.is_empty() {
                    searcher.search(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        mode,
                    )
                } else {
                    searcher.search_with_start_points(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        active_seeds,
                        mode,
                    )
                };

                match result {
"""
    new_call = """                let active_seeds: &[u32] = if !global_start_ids.is_empty() {
                    global_start_ids.as_slice()
                } else {
                    seeds.as_slice()
                };
                let result = if let Some(index) = hint_ivf.as_ref() {
                    searcher.search_with_hint_ivf(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        &index.medoid_ids,
                        &index.offsets,
                        &index.hint_ids,
                        hint_ivf_nprobe,
                    )
                } else if active_seeds.is_empty() {
                    searcher.search(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        mode,
                    )
                } else {
                    searcher.search_with_start_points(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        active_seeds,
                        mode,
                    )
                };

                match result {
"""
    s = once(s, old_call, new_call, "integrated Hint-IVF benchmark dispatch")

    marker = """fn load_global_start_ids(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
"""
    helper = r'''struct HintIvfIndex {
    medoid_ids: Vec<u32>,
    offsets: Vec<u32>,
    hint_ids: Vec<u32>,
}

impl HintIvfIndex {
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GHIVF001";
        if raw.len() < 24 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid Hint-IVF header");
        }
        let nlist = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let child_count = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let total_landmarks = u32::from_le_bytes(raw[16..20].try_into()?) as usize;
        let reserved = u32::from_le_bytes(raw[20..24].try_into()?);
        if nlist == 0 || child_count + nlist != total_landmarks || reserved != 0 {
            anyhow::bail!("invalid Hint-IVF shape");
        }

        let expected = 24usize
            .checked_add(nlist.checked_mul(4).ok_or_else(|| anyhow::anyhow!("Hint-IVF overflow"))?)
            .and_then(|x| x.checked_add((nlist + 1).checked_mul(4)?))
            .and_then(|x| x.checked_add(child_count.checked_mul(4)?))
            .ok_or_else(|| anyhow::anyhow!("Hint-IVF size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "Hint-IVF byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }

        let mut off = 24usize;
        let mut medoid_ids = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            medoid_ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        let mut offsets = Vec::with_capacity(nlist + 1);
        for _ in 0..=nlist {
            offsets.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(child_count as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            anyhow::bail!("invalid Hint-IVF offsets");
        }

        let mut hint_ids = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            hint_ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if off != raw.len() {
            anyhow::bail!("Hint-IVF parser did not consume all bytes");
        }

        let mut unique = HashSet::with_capacity(total_landmarks);
        for id in medoid_ids.iter().chain(hint_ids.iter()) {
            if !unique.insert(*id) {
                anyhow::bail!("duplicate ID in Hint-IVF");
            }
        }

        Ok(Self {
            medoid_ids,
            offsets,
            hint_ids,
        })
    }
}

'''
    s = once(s, marker, helper + marker, "optimized Hint-IVF parser")
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
    patch_provider_medoid(provider)
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf_fast(provider)
    patch_benchmark_hint_ivf_fast(benchmark)
    print(f"patched DiskANN {PINNED} with optimized ID-only Hint-IVF")


if __name__ == "__main__":
    main()
