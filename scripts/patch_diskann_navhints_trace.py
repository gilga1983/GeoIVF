#!/usr/bin/env python3
"""Patch optimized NavHints with traversal recording for on-policy training."""
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
        "trace field",
    )

    # Generic search results do not need traces in this specialized training
    # binary, but every constructor must initialize the added field.
    generic = """        Ok(SearchResultStats {
            cmps: query_stats.total_comparisons,
            result_count: stats.result_count,
            query_statistics: query_stats.clone(),
        })
"""
    if generic in s:
        s = s.replace(
            generic,
            """        Ok(SearchResultStats {
            cmps: query_stats.total_comparisons,
            result_count: stats.result_count,
            query_statistics: query_stats.clone(),
            trace_ids: Vec::new(),
        })
""",
            1,
        )

    old = """        let stats = self.runtime.block_on(self.index.search(
            knn_search,
            &strategy,
            &DefaultContext,
            query,
            &mut result_output_buffer,
        ))?;

        let routing_comparisons = io_tracker
"""
    new = """        let trace_enabled = std::env::var_os("DISKANN_TRACE_FILE").is_some();
        let (stats, trace_ids) = if trace_enabled {
            let mut record = VisitedSearchRecord::new((search_list_size as usize).max(16));
            let recorded = RecordedKnn::new(knn_search, &mut record);
            let stats = self.runtime.block_on(self.index.search(
                recorded,
                &strategy,
                &DefaultContext,
                query,
                &mut result_output_buffer,
            ))?;
            (stats, record.ids().collect::<Vec<u32>>())
        } else {
            let stats = self.runtime.block_on(self.index.search(
                knn_search,
                &strategy,
                &DefaultContext,
                query,
                &mut result_output_buffer,
            ))?;
            (stats, Vec::new())
        };

        let routing_comparisons = io_tracker
"""
    s = once(s, old, new, "record optimized NavHints traversal")

    old_result = """            stats: SearchResultStats {
                cmps: query_stats.total_comparisons,
                result_count: stats.result_count,
                query_statistics: query_stats,
            },
"""
    new_result = """            stats: SearchResultStats {
                cmps: query_stats.total_comparisons,
                result_count: stats.result_count,
                query_statistics: query_stats,
                trace_ids,
            },
"""
    s = once(s, old_result, new_result, "attach optimized trace")

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
        "trace Write import",
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
""",
        """            .zip(statistics_vec.par_iter_mut())
            .zip(trace_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut());
""",
        "zip trace vector",
    )

    # The optimized benchmark closure shape after patch_benchmark_global.
    old_closure = """            |(((((q, vf), id_chunk), dist_chunk), stats), rc)| {
"""
    new_closure = """            |((((((q, vf), id_chunk), dist_chunk), stats), trace_out), rc)| {
"""
    s = once(s, old_closure, new_closure, "trace closure shape")

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
        "capture trace IDs",
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
    s = once(s, anchor, dump + anchor, "dump on-policy traversal trace")
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

    expose_pq_batch_lookup(root)
    patch_provider(provider)
    patch_provider_medoid(provider)
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf_fast(provider)
    patch_benchmark_hint_ivf_fast(benchmark)
    patch_provider_trace(provider)
    patch_benchmark_trace(benchmark)
    print(f"patched DiskANN {PINNED} for on-policy NavHints trace collection")


if __name__ == "__main__":
    main()
