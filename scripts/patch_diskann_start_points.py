#!/usr/bin/env python3
"""Patch pinned DiskANN3 to accept per-query custom graph start points.

The patch changes only query initialization and the benchmark harness. Vamana,
PQ codes, graph traversal, disk records, full-precision reranking, and provider
I/O remain upstream. Every replacement is exact and fails closed on drift.
"""
from __future__ import annotations
import argparse
from pathlib import Path

PINNED = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"

def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)

def patch_provider(path: Path) -> None:
    s = path.read_text()

    s = once(s,
"""    /// Scratch pool for disk search operations that need allocations.
    scratch_pool: &'a Arc<ObjectPool<DiskSearchScratch<Data, ProviderFactory::VertexProviderType>>>,
}
""",
"""    /// Scratch pool for disk search operations that need allocations.
    scratch_pool: &'a Arc<ObjectPool<DiskSearchScratch<Data, ProviderFactory::VertexProviderType>>>,

    /// Optional query-specific graph entry points. None preserves the upstream medoid.
    start_points: Option<&'a [u32]>,
}
""", "strategy field")

    s = once(s,
"""    cache_indexed_vectors: bool,
    flat_scan_filter: Option<IdFilter<'a>>,
}
""",
"""    cache_indexed_vectors: bool,
    flat_scan_filter: Option<IdFilter<'a>>,
    start_points: Option<&'a [u32]>,
}
""", "accessor field")

    s = once(s,
"""    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![start_vertex_id])
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }
""",
"""    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        if let Some(ids) = self.start_points {
            return Ok(ids.to_vec());
        }
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![start_vertex_id])
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if let Some(ids) = self.start_points {
            return self.pq_distances(ids, |dist, id| f(id, dist));
        }
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }
""", "start points accessor")

    s = once(s,
"""            cache_indexed_vectors: strategy.cache_indexed_vectors,
            flat_scan_filter: None,
        })
""",
"""            cache_indexed_vectors: strategy.cache_indexed_vectors,
            flat_scan_filter: None,
            start_points: strategy.start_points,
        })
""", "accessor constructor")

    old_strategy = """    fn search_strategy<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        postprocess_filter: PostprocessStrategy<'a>,
        cache_indexed_vectors: bool,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors,
            postprocess_filter,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
        }
    }
"""
    new_strategy = """    fn search_strategy<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        postprocess_filter: PostprocessStrategy<'a>,
        cache_indexed_vectors: bool,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        self.search_strategy_with_start_points(
            io_tracker,
            postprocess_filter,
            cache_indexed_vectors,
            None,
        )
    }

    fn search_strategy_with_start_points<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        postprocess_filter: PostprocessStrategy<'a>,
        cache_indexed_vectors: bool,
        start_points: Option<&'a [u32]>,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors,
            postprocess_filter,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points,
        }
    }
"""
    s = once(s, old_strategy, new_strategy, "strategy constructor")

    marker = """    /// Perform a search on the disk index and return each result with its native indexed vector.
"""
    seeded_method = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
    ///
    /// This changes only the initial candidate set. The same graph traversal,
    /// disk provider, full-precision reranking, and statistics path are used.
    pub fn search_with_start_points(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        start_points: &[u32],
        mode: SearchMode<'_>,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if start_points.is_empty() {
            return Err(diskann_error!(ErrorKind::IndexError, "start_points must not be empty"));
        }
        let num_points = self.index.provider().num_points;
        for (i, id) in start_points.iter().enumerate() {
            if (*id as usize) >= num_points {
                return Err(diskann_error!(ErrorKind::IndexError, "start point out of range"));
            }
            if start_points[..i].contains(id) {
                return Err(diskann_error!(ErrorKind::IndexError, "duplicate start point"));
            }
        }

        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; return_list_size as usize];
        let mut distances = vec![0f32; return_list_size as usize];
        let mut associated_data =
            vec![Data::AssociatedDataType::default(); return_list_size as usize];

        if search_list_size < return_list_size {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "search list size must be at least as large as the number of results requested",
            ));
        }

        let stats = self.search_internal_impl(
            query,
            return_list_size as usize,
            search_list_size,
            beam_width,
            &mut query_stats,
            &mut indices,
            &mut distances,
            &mut associated_data,
            None,
            Some(start_points),
            &mode,
        )?;

        let mut search_result = SearchResult {
            results: Vec::with_capacity(return_list_size as usize),
            stats,
        };
        for ((vertex_id, distance), associated_data) in
            indices.into_iter().zip(distances).zip(associated_data)
        {
            search_result.results.push(SearchResultItem {
                vertex_id,
                distance,
                data: associated_data,
            });
        }
        Ok(search_result)
    }

"""
    s = once(s, marker, seeded_method + marker, "seeded public method")

    s = once(s,
"""            Some(&mut indexed_vectors),
            &mode,
""",
"""            Some(&mut indexed_vectors),
            None,
            &mode,
""", "indexed-vector call")

    s = once(s,
"""            associated_data,
            None,
            mode,
        )
""",
"""            associated_data,
            None,
            None,
            mode,
        )
""", "raw search call")

    s = once(s,
"""        associated_data: &mut [Data::AssociatedDataType],
        indexed_vectors: IndexedVectorOutput<'_, Data::VectorDataType>,
        mode: &SearchMode<'_>,
""",
"""        associated_data: &mut [Data::AssociatedDataType],
        indexed_vectors: IndexedVectorOutput<'_, Data::VectorDataType>,
        start_points: Option<&[u32]>,
        mode: &SearchMode<'_>,
""", "impl signature")

    # Only mutate strategy construction inside search_internal_impl.
    head, sep, tail = s.partition("    fn search_internal_impl(")
    if not sep:
        raise RuntimeError("search_internal_impl not found")
    body, sep2, rest = tail.partition("        query_stats.total_comparisons = stats.cmps;")
    if not sep2:
        raise RuntimeError("search_internal_impl end marker not found")
    body = body.replace("self.search_strategy(", "self.search_strategy_with_start_points(")
    body = body.replace(
"""                    cache_indexed_vectors,
                );""",
"""                    cache_indexed_vectors,
                    start_points,
                );""")
    body = body.replace(
"""self.search_strategy_with_start_points(&io_tracker, postprocess_config, cache_indexed_vectors);""",
"""self.search_strategy_with_start_points(
                        &io_tracker,
                        postprocess_config,
                        cache_indexed_vectors,
                        start_points,
                    );""")
    s = head + sep + body + sep2 + rest

    if s.count("search_with_start_points(") != 1:
        raise RuntimeError("unexpected seeded method count")
    path.write_text(s)

def patch_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(s,
"""    let num_queries = queries.nrows();

    // Load the vector filters
""",
"""    let num_queries = queries.nrows();

    // Optional fixed-width query-specific graph seeds. Absence exactly preserves
    // the released medoid-start behavior.
    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };

    // Load the vector filters
""", "benchmark seed load")

    s = once(s,
"""            .zip(statistics_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
""",
"""            .zip(statistics_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut())
            .zip(seed_rows.par_iter());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |((((((q, vf), id_chunk), dist_chunk), stats), rc), seeds)| {
""", "benchmark zip")

    old_call = """                match searcher.search(
                    q,
                    search_params.recall_at,
                    l,
                    Some(search_params.beam_width),
                    mode,
                ) {
"""
    new_call = """                let result = if seeds.is_empty() {
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
                        seeds,
                        mode,
                    )
                };

                match result {
"""
    s = once(s, old_call, new_call, "benchmark seeded call")

    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''fn load_start_points(path: &std::path::Path, expected_rows: usize) -> anyhow::Result<Vec<Vec<u32>>> {
    let raw = std::fs::read(path)?;
    if raw.len() < 8 {
        anyhow::bail!("start-point file is shorter than its header");
    }
    let rows = u32::from_le_bytes(raw[0..4].try_into()?) as usize;
    let cols = u32::from_le_bytes(raw[4..8].try_into()?) as usize;
    if rows != expected_rows || cols == 0 {
        anyhow::bail!(
            "start-point shape mismatch: got {}x{}, expected {}xpositive",
            rows,
            cols,
            expected_rows
        );
    }
    let expected = 8usize
        .checked_add(rows.checked_mul(cols).and_then(|n| n.checked_mul(4)).ok_or_else(|| {
            anyhow::anyhow!("start-point size overflow")
        })?)
        .ok_or_else(|| anyhow::anyhow!("start-point size overflow"))?;
    if raw.len() != expected {
        anyhow::bail!("start-point file byte length mismatch");
    }
    let mut result = Vec::with_capacity(rows);
    let mut offset = 8;
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        let mut unique = HashSet::with_capacity(cols);
        for _ in 0..cols {
            let id = u32::from_le_bytes(raw[offset..offset + 4].try_into()?);
            offset += 4;
            if !unique.insert(id) {
                anyhow::bail!("duplicate start point in one query row");
            }
            row.push(id);
        }
        result.push(row);
    }
    Ok(result)
}

'''
    s = once(s, marker, helper + marker, "benchmark seed parser")
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
    print(f"patched {PINNED} for query-specific graph start points")

if __name__ == "__main__":
    main()
