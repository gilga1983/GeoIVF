#!/usr/bin/env python3
"""Patch the pinned DiskANN SSD benchmark to record graph-expansion traces.

Applies the current paper-Catapult + static-IP-portal patch first, then enables
an opt-in traversal recorder. With DISKANN_TRACE_FILE set, ordinary graph search
uses DiskANN's native RecordedKnn/VisitedSearchRecord and writes one JSONL row
per query after the parallel search completes. Each row contains query index,
expanded vertex IDs in order, result ID, and measured I/O count.

The trace is analytical only: search behavior is otherwise unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_paper_catapult import (
    PINNED,
    patch_benchmark as patch_catapult_benchmark,
    patch_provider_medoid,
)
from patch_diskann_start_points import once, patch_provider


def patch_provider_trace(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """        search::{AdaptiveL, InlineFilterSearch, Knn},
""",
        """        search::{
            record::VisitedSearchRecord, AdaptiveL, InlineFilterSearch, Knn, RecordedKnn,
        },
""",
        "recorded knn imports",
    )

    s = once(
        s,
        """pub struct SearchResultStats {
    pub cmps: u32,
    pub result_count: u32,
    pub query_statistics: QueryStatistics,
}
""",
        """pub struct SearchResultStats {
    pub cmps: u32,
    pub result_count: u32,
    pub query_statistics: QueryStatistics,
    pub trace_ids: Vec<u32>,
}
""",
        "trace field in result stats",
    )

    s = once(
        s,
        """        let stats = match mode {
""",
        """        let trace_enabled = std::env::var_os("DISKANN_TRACE_FILE").is_some();
        let mut trace_ids = Vec::<u32>::new();

        let stats = match mode {
""",
        "trace setup",
    )

    old_graph = """            SearchMode::Graph { filter } => {
                let strategy = self.search_strategy(
                    &io_tracker,
                    filter
                        .as_deref()
                        .map_or(PostprocessStrategy::AcceptAll, PostprocessStrategy::Apply),
                    cache_indexed_vectors,
                );
                let knn_search = Knn::new(l, beam_width)
                    .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
                self.runtime.block_on(self.index.search(
                    knn_search,
                    &strategy,
                    &DefaultContext,
                    query,
                    &mut result_output_buffer,
                ))?
            }
"""
    new_graph = """            SearchMode::Graph { filter } => {
                let strategy = self.search_strategy(
                    &io_tracker,
                    filter
                        .as_deref()
                        .map_or(PostprocessStrategy::AcceptAll, PostprocessStrategy::Apply),
                    cache_indexed_vectors,
                );
                let knn_search = Knn::new(l, beam_width)
                    .map_err(|e| diskann_error!(ErrorKind::IndexError, e))?;
                if trace_enabled {
                    let mut record = VisitedSearchRecord::new(l.max(16));
                    let recorded = RecordedKnn::new(knn_search, &mut record);
                    let stats = self.runtime.block_on(self.index.search(
                        recorded,
                        &strategy,
                        &DefaultContext,
                        query,
                        &mut result_output_buffer,
                    ))?;
                    trace_ids.extend(record.ids());
                    stats
                } else {
                    self.runtime.block_on(self.index.search(
                        knn_search,
                        &strategy,
                        &DefaultContext,
                        query,
                        &mut result_output_buffer,
                    ))?
                }
            }
"""
    s = once(s, old_graph, new_graph, "record graph expansion path")

    s = once(
        s,
        """        Ok(SearchResultStats {
            cmps: query_stats.total_comparisons,
            result_count: stats.result_count,
            query_statistics: query_stats.clone(),
        })
""",
        """        Ok(SearchResultStats {
            cmps: query_stats.total_comparisons,
            result_count: stats.result_count,
            query_statistics: query_stats.clone(),
            trace_ids,
        })
""",
        "return trace ids",
    )
    path.write_text(s)


def patch_benchmark_trace(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use std::{
    collections::{HashSet, VecDeque},
    fmt,
""",
        """use std::{
    collections::{HashSet, VecDeque},
    fmt,
    io::Write,
""",
        "trace write import",
    )

    s = once(
        s,
        """        let mut statistics_vec: Vec<QueryStatistics> =
            vec![QueryStatistics::default(); num_queries];
        let mut result_counts: Vec<u32> = vec![0; num_queries];
""",
        """        let mut statistics_vec: Vec<QueryStatistics> =
            vec![QueryStatistics::default(); num_queries];
        let mut trace_vec: Vec<Vec<u32>> =
            (0..num_queries).map(|_| Vec::<u32>::new()).collect();
        let mut result_counts: Vec<u32> = vec![0; num_queries];
""",
        "allocate trace vectors",
    )

    s = once(
        s,
        """            .zip(statistics_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
""",
        """            .zip(statistics_vec.par_iter_mut())
            .zip(trace_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |((((((q, vf), id_chunk), dist_chunk), stats), trace_out), rc)| {
""",
        "zip per-query trace output",
    )

    s = once(
        s,
        """                    Ok(search_result) => {
                        *stats = search_result.stats.query_statistics;
                        let base_count = (search_result.stats.result_count as usize)
""",
        """                    Ok(mut search_result) => {
                        *trace_out = std::mem::take(&mut search_result.stats.trace_ids);
                        *stats = search_result.stats.query_statistics.clone();
                        let base_count = (search_result.stats.result_count as usize)
""",
        "capture trace output",
    )

    anchor = """        let total_time = start.elapsed();
"""
    dump = r'''        if let Some(base_path) = std::env::var_os("DISKANN_TRACE_FILE") {
            let base = std::path::PathBuf::from(base_path);
            let path = if search_params.search_list.len() == 1 {
                base
            } else {
                let mut p = base.clone();
                let ext = p
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("jsonl")
                    .to_string();
                p.set_extension(format!("L{}.{}", l, ext));
                p
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::File::create(&path)?;
            let mut out = std::io::BufWriter::new(file);
            for qi in 0..num_queries {
                let result_id = if result_counts[qi] > 0 {
                    result_ids[qi * search_params.recall_at as usize]
                } else {
                    u32::MAX
                };
                let ids = &trace_vec[qi];
                write!(
                    out,
                    "{{\"query\":{},\"result_id\":{},\"io_count\":{},\"ids\":[",
                    qi,
                    result_id,
                    statistics_vec[qi].total_io_operations
                )?;
                for (j, id) in ids.iter().enumerate() {
                    if j > 0 {
                        write!(out, ",")?;
                    }
                    write!(out, "{}", id)?;
                }
                writeln!(out, "]}}")?;
            }
            out.flush()?;
        }

'''
    s = once(s, anchor, dump + anchor, "dump traversal trace")
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
    patch_catapult_benchmark(benchmark)
    patch_provider_trace(provider)
    patch_benchmark_trace(benchmark)
    print(f"patched DiskANN {PINNED} for traversal trace collection")


if __name__ == "__main__":
    main()
