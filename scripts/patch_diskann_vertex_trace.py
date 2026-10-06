#!/usr/bin/env python3
"""Patch vertex-NavHints DiskANN to dump exact expanded-vertex traces.

Apply after patch_diskann_vertex_navhints.py. When DISKANN_VERTEX_TRACE_FILE is
set, search_with_vertex_hint_ivf records the exact traversal via RecordedKnn and
returns those IDs in SearchResultStats. The benchmark writes one JSONL record
per query.

This is a research-only tracing binary. Normal search behavior is unchanged
when the environment variable is absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s=path.read_text()

    s=once(
        s,
        """        search::{AdaptiveL, InlineFilterSearch, Knn},
""",
        """        search::{
            record::VisitedSearchRecord, AdaptiveL, InlineFilterSearch, Knn, RecordedKnn,
        },
""",
        "vertex trace imports",
    )

    s=once(
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
        "vertex trace stats field",
    )

    # Base generic search constructor.
    s=once(
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
            trace_ids: Vec::new(),
        })
""",
        "generic trace field init",
    )

    old=r'''        let stats = self.runtime.block_on(self.index.search(
            knn_search,
            &strategy,
            &DefaultContext,
            query,
            &mut result_output_buffer,
        ))?;

        let routing_comparisons = io_tracker
'''
    new=r'''        let trace_enabled = std::env::var_os("DISKANN_VERTEX_TRACE_FILE").is_some();
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
'''
    # Vertex patch adds exactly one copy of this search block after the base
    # method has already been patched by the fast Hint-IVF machinery.
    if s.count(old) != 1:
        raise RuntimeError(f"vertex trace search block expected once, found {s.count(old)}")
    s=s.replace(old,new,1)

    old_stats=r'''            stats: SearchResultStats {
                cmps: query_stats.total_comparisons,
                result_count: stats.result_count,
                query_statistics: query_stats,
            },
'''
    new_stats=r'''            stats: SearchResultStats {
                cmps: query_stats.total_comparisons,
                result_count: stats.result_count,
                query_statistics: query_stats,
                trace_ids,
            },
'''
    if s.count(old_stats) != 1:
        raise RuntimeError(f"vertex result stats expected once, found {s.count(old_stats)}")
    s=s.replace(old_stats,new_stats,1)
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s=path.read_text()
    s=once(s,"""use rayon::prelude::*;
""","""use rayon::prelude::*;
use std::io::Write;
""","vertex trace Write import")

    s=once(
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
        "allocate vertex trace vectors",
    )

    s=once(
        s,
        """            .zip(statistics_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut())
            .zip(seed_rows.par_iter());
""",
        """            .zip(statistics_vec.par_iter_mut())
            .zip(trace_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut())
            .zip(seed_rows.par_iter());
""",
        "zip vertex trace vector",
    )

    s=once(
        s,
        """            |((((((q, vf), id_chunk), dist_chunk), stats), rc), seeds)| {
""",
        """            |(((((((q, vf), id_chunk), dist_chunk), stats), trace_out), rc), seeds)| {
""",
        "vertex trace closure",
    )

    # All relevant result-returning search paths now have trace_ids. Only the
    # vertex-hint path populates them; other paths leave them empty.
    s=once(
        s,
        """                    Ok(search_result) => {
                        *stats = search_result.stats.query_statistics;
""",
        """                    Ok(mut search_result) => {
                        *trace_out = std::mem::take(&mut search_result.stats.trace_ids);
                        *stats = search_result.stats.query_statistics;
""",
        "capture vertex trace",
    )

    anchor="""        let total_time = start.elapsed();
"""
    dump=r'''        if let Some(base_path) = std::env::var_os("DISKANN_VERTEX_TRACE_FILE") {
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
                write!(
                    out,
                    "{{\"query\":{},\"result_id\":{},\"io_count\":{},\"ids\":[",
                    qi,
                    result_id,
                    statistics_vec[qi].total_io_operations
                )?;
                for (j, id) in trace_vec[qi].iter().enumerate() {
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
    s=once(s,anchor,dump+anchor,"dump exact vertex trace")
    path.write_text(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("diskann",type=Path);args=ap.parse_args()
    root=args.diskann.resolve()
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("patched vertex-NavHints DiskANN with exact traversal tracing")


if __name__=="__main__":main()
