#!/usr/bin/env python3
"""Patch pinned DiskANN3 for integrated portals plus per-query audit output.

The audit hook is enabled only when DISKANN_QUERY_AUDIT_FILE is set and is
intended for offline portal training. Ordinary benchmark behavior is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_integrated_portals import patch_integrated_benchmark
from patch_diskann_start_points import PINNED, once, patch_benchmark, patch_provider


def patch_query_audit(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    let num_queries = queries.nrows();

    // Optional fixed-width query-specific graph seeds. Absence exactly preserves
""",
        """    let num_queries = queries.nrows();

    // Optional offline per-query audit dump. Restricting audit mode to a single
    // L keeps the output unambiguous and avoids accidental overwrite.
    let query_audit_path =
        std::env::var_os("DISKANN_QUERY_AUDIT_FILE").map(std::path::PathBuf::from);
    if query_audit_path.is_some() && search_params.search_list.len() != 1 {
        anyhow::bail!("DISKANN_QUERY_AUDIT_FILE requires exactly one search L");
    }

    // Optional fixed-width query-specific graph seeds. Absence exactly preserves
""",
        "query audit setup",
    )

    s = once(
        s,
        """        l_span.end();
        search_results_per_l.push(search_result);
""",
        """        if let Some(path) = query_audit_path.as_ref() {
            let width = search_params.recall_at as usize;
            let rows = statistics_vec
                .iter()
                .zip(result_counts.iter())
                .zip(result_ids.chunks_exact(width))
                .enumerate()
                .map(|(query_id, ((stats, count), ids))| {
                    let written = (*count as usize).min(width);
                    serde_json::json!({
                        "query_id": query_id,
                        "total_execution_time_us": stats.total_execution_time_us as u64,
                        "io_time_us": stats.io_time_us as u64,
                        "cpu_time_us": stats.cpu_time_us as u64,
                        "total_io_operations": stats.total_io_operations,
                        "total_comparisons": stats.total_comparisons,
                        "total_vertices_loaded": stats.total_vertices_loaded,
                        "search_hops": stats.search_hops,
                        "result_ids": ids[..written].to_vec(),
                    })
                })
                .collect::<Vec<_>>();
            let audit = serde_json::json!({
                "search_l": l,
                "rows": rows,
            });
            std::fs::write(path, serde_json::to_vec(&audit)?)?;
        }

        l_span.end();
        search_results_per_l.push(search_result);
""",
        "query audit dump",
    )

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
    patch_integrated_benchmark(benchmark)
    patch_query_audit(benchmark)
    print(f"patched {PINNED} for integrated navigation-portal training")


if __name__ == "__main__":
    main()
