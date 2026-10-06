#!/usr/bin/env python3
"""Add a real online semantic result cache to vertex-NavHint DiskANN benchmark.

Apply after patch_diskann_vertex_navhints.py.

When DISKANN_SEMANTIC_CACHE_CAPACITY > 0:
* queries are processed sequentially in workload order;
* prior query vectors/results are kept in a rolling exact-scan cache;
* query vectors are stored as f32 for this first native implementation;
* the nearest cached query by inner product contributes its first N actual
  DiskANN result IDs as additional starting points;
* lookup time is charged to query CPU/latency;
* the first W queries warm the cache and are excluded from reported metrics.

The normal parallel benchmark path is unchanged when the cache is disabled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_start_points import once


def section_replace(s: str, begin: str, end: str, old: str, new: str, label: str) -> str:
    a = s.find(begin)
    if a < 0:
        raise RuntimeError(f"{label}: begin marker missing")
    b = s.find(end, a)
    if b < 0:
        raise RuntimeError(f"{label}: end marker missing")
    chunk = s[a:b]
    n = chunk.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one match in section, found {n}")
    chunk = chunk.replace(old, new, 1)
    return s[:a] + chunk + s[b:]


def patch_provider(path: Path) -> None:
    s = path.read_text()

    old_struct = """#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
}
"""
    new_struct = """#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
    medoid_ids: &'a [u32],
    coarse_local_ids: &'a [u32],
    coarse_pq_codes: &'a [u8],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    nprobe: usize,
    extra_start_points: Option<&'a [u32]>,
}
"""
    s = once(s, old_struct, new_struct, "semantic extra starts field")

    # Baseline Hint-IVF constructor remains unchanged semantically.
    s = section_replace(
        s,
        "    pub fn search_with_hint_ivf(",
        "    /// Perform the ordinary graph search from caller-supplied starting vertices.",
        """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
        };
""",
        """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            extra_start_points: None,
        };
""",
        "baseline Hint-IVF constructor",
    )

    # Vertex path accepts optional semantic-cache result IDs.
    s = section_replace(
        s,
        "    pub fn search_with_vertex_hint_ivf(",
        "    /// Perform the ordinary graph search from caller-supplied starting vertices.",
        """        nprobe: usize,
        vertex_hint_variant: usize,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
""",
        """        nprobe: usize,
        vertex_hint_variant: usize,
        extra_start_points: Option<&[u32]>,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
""",
        "vertex method semantic arg",
    )
    s = section_replace(
        s,
        "    pub fn search_with_vertex_hint_ivf(",
        "    /// Perform the ordinary graph search from caller-supplied starting vertices.",
        """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
        };
""",
        """        let hint_ivf = HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            extra_start_points,
        };
""",
        "vertex Hint-IVF constructor",
    )

    old_emit = """            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
"""
    new_emit = """            // Optionally seed the same initial frontier with successful IDs
            // from the nearest prior semantic-cache query. All IDs are re-valued
            // with the current query's existing PQ lookup table.
            let mut emitted = 1usize;
            if let Some(extra) = ivf.extra_start_points {
                routing_cmps = routing_cmps.checked_add(extra.len()).ok_or_else(|| {
                    diskann_error!(ErrorKind::IndexError, "semantic-cache comparison overflow")
                })?;
                self.pq_distances(extra, |distance, id| {
                    if id != winner.1 {
                        emitted += 1;
                        f(id, distance);
                    }
                })?;
            }

            // Search itself counts emitted starts. Add only selector/cache PQ
            // comparisons that are not already represented there.
            self.io_tracker.routing_comparisons.fetch_add(
                routing_cmps.saturating_sub(emitted),
                std::sync::atomic::Ordering::Relaxed,
            );
            f(winner.1, winner.0);
            return Ok(());
"""
    s = once(s, old_emit, new_emit, "emit semantic cache starts")

    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, time::Instant};
""",
        """use std::{
    collections::{HashSet, VecDeque},
    fmt,
    sync::atomic::AtomicBool,
    time::Instant,
};
""",
        "semantic cache imports",
    )

    anchor = """    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }

    // Load the vector filters
"""
    config = """    if vertex_hint_variant.is_some() && hint_ivf.is_none() {
        anyhow::bail!("vertex hints require DISKANN_HINT_IVF_FILE");
    }
    let semantic_cache_capacity = std::env::var("DISKANN_SEMANTIC_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let semantic_cache_results = std::env::var("DISKANN_SEMANTIC_CACHE_RESULTS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    let semantic_cache_warmup = std::env::var("DISKANN_SEMANTIC_CACHE_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(500);
    if semantic_cache_capacity > 0 {
        if hint_ivf.is_none() || vertex_hint_variant.is_none() {
            anyhow::bail!("semantic cache requires Hint-IVF and vertex hints");
        }
        if semantic_cache_results == 0 || semantic_cache_results > search_params.recall_at as usize {
            anyhow::bail!("semantic-cache result count outside valid range");
        }
        if semantic_cache_warmup >= num_queries {
            anyhow::bail!("semantic-cache warmup must leave measured queries");
        }
        if search_params.vector_filters_file.is_some() {
            anyhow::bail!("semantic-cache screen does not support vector filters");
        }
    }

    // Load the vector filters
"""
    s = once(s, anchor, config, "semantic cache config")

    # Existing parallel vertex path passes no semantic-cache starts.
    old_call = """                            hint_ivf_nprobe,
                            variant,
                        )
"""
    new_call = """                            hint_ivf_nprobe,
                            variant,
                            None,
                        )
"""
    n = s.count(old_call)
    if n != 1:
        raise RuntimeError(f"parallel vertex call: expected one match, found {n}")
    s = s.replace(old_call, new_call, 1)

    # Install helper structure before GroundTruthContext.
    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''struct SemanticCacheEntry {
    query: Box<[f32]>,
    results: Vec<u32>,
}

fn slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("semantic-cache GT slicing does not support filtered truth");
    }
    let ids = ctx
        .gt_ids
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GT IDs missing"))?;
    if start > ids.len() / ctx.gt_dim {
        anyhow::bail!("semantic-cache GT slice outside rows");
    }
    let off = start * ctx.gt_dim;
    Ok(GroundTruthContext {
        gt_ids: Some(ids[off..].to_vec()),
        gt_ids_variable_length: None,
        gt_dists: ctx.gt_dists.as_ref().map(|x| x[off..].to_vec()),
        gt_dim: ctx.gt_dim,
        recall_at: ctx.recall_at,
    })
}

'''
    s = once(s, marker, helper + marker, "semantic cache helpers")

    # Replace the query-parallel block with an optional sequential online path.
    start_marker = "        let zipped = queries\n"
    end_marker = "        let total_time = start.elapsed();\n"
    a = s.find(start_marker)
    if a < 0:
        raise RuntimeError("semantic cache: zipped block start missing")
    b = s.find(end_marker, a)
    if b < 0:
        raise RuntimeError("semantic cache: zipped block end missing")
    parallel_block = s[a:b]
    sequential = r'''        if semantic_cache_capacity > 0 {
            let index = hint_ivf
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("semantic cache requires Hint-IVF"))?;
            let variant = vertex_hint_variant
                .ok_or_else(|| anyhow::anyhow!("semantic cache requires vertex hints"))?;
            let mut semantic_cache = VecDeque::<SemanticCacheEntry>::new();

            for qi in 0..num_queries {
                let q = queries.row(qi);
                let vf = &vector_filters[qi];
                let has_filter = search_params.vector_filters_file.is_some();
                let mode: SearchMode<'_> = search_params.search_mode.search_mode(
                    has_filter,
                    vf,
                    search_params.post_processor.as_ref(),
                );

                let scan_start = Instant::now();
                let qf = T::as_f32(q)
                    .map_err(|e| anyhow::anyhow!("semantic cache query conversion failed: {e}"))?;
                let mut semantic_starts = Vec::<u32>::new();
                if qi >= semantic_cache_warmup && !semantic_cache.is_empty() {
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_entry: Option<&SemanticCacheEntry> = None;
                    for entry in semantic_cache.iter() {
                        let mut score = 0.0f32;
                        for (&a, &b) in qf.iter().zip(entry.query.iter()) {
                            score = a.mul_add(b, score);
                        }
                        if score > best_score {
                            best_score = score;
                            best_entry = Some(entry);
                        }
                    }
                    if let Some(entry) = best_entry {
                        semantic_starts.extend(
                            entry.results.iter().take(semantic_cache_results).copied(),
                        );
                    }
                }
                let scan_us = scan_start.elapsed().as_micros();

                let extra = if semantic_starts.is_empty() {
                    None
                } else {
                    Some(semantic_starts.as_slice())
                };
                let mut search_result = searcher.search_with_vertex_hint_ivf(
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
                    extra,
                )?;

                search_result.stats.query_statistics.total_execution_time_us =
                    search_result
                        .stats
                        .query_statistics
                        .total_execution_time_us
                        .saturating_add(scan_us);
                search_result.stats.query_statistics.cpu_time_us =
                    search_result
                        .stats
                        .query_statistics
                        .cpu_time_us
                        .saturating_add(scan_us);

                let base_count = (search_result.stats.result_count as usize)
                    .min(search_params.recall_at as usize)
                    .min(search_result.results.len());
                let base = qi * search_params.recall_at as usize;
                let id_chunk =
                    &mut result_ids[base..base + search_params.recall_at as usize];
                let dist_chunk =
                    &mut result_dists[base..base + search_params.recall_at as usize];
                id_chunk.fill(0);
                dist_chunk.fill(0.0);
                result_counts[qi] = base_count as u32;
                for (i, item) in search_result.results.iter().take(base_count).enumerate() {
                    id_chunk[i] = item.vertex_id;
                    dist_chunk[i] = item.distance;
                }
                statistics_vec[qi] = search_result.stats.query_statistics;

                let cached_results = search_result
                    .results
                    .iter()
                    .take(base_count)
                    .map(|item| item.vertex_id)
                    .collect::<Vec<_>>();
                semantic_cache.push_back(SemanticCacheEntry {
                    query: qf.to_vec().into_boxed_slice(),
                    results: cached_results,
                });
                while semantic_cache.len() > semantic_cache_capacity {
                    semantic_cache.pop_front();
                }
            }
        } else {
''' + parallel_block + r'''        }
'''
    s = s[:a] + sequential + s[b:]

    # Report metrics only on the measured suffix when semantic cache is enabled.
    old_result = """        let search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;
"""
    new_result = """        let search_result = if semantic_cache_capacity > 0 {
            let start_q = semantic_cache_warmup;
            let result_offset = start_q * search_params.recall_at as usize;
            let sliced_gt = slice_ground_truth_context(&gt_context, start_q)?;
            DiskSearchResult::new(
                &statistics_vec[start_q..],
                &result_ids[result_offset..],
                &result_counts[start_q..],
                l,
                total_time.as_secs_f32(),
                num_queries - start_q,
                &sliced_gt,
            )?
        } else {
            DiskSearchResult::new(
                &statistics_vec,
                &result_ids,
                &result_counts,
                l,
                total_time.as_secs_f32(),
                num_queries,
                &gt_context,
            )?
        };
"""
    s = once(s, old_result, new_result, "semantic measured suffix")

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    provider = root / "diskann-disk/src/search/provider/disk_provider.rs"
    benchmark = root / "diskann-benchmark/src/disk_index/search.rs"
    if not provider.is_file() or not benchmark.is_file():
        raise SystemExit("unexpected DiskANN checkout")
    patch_provider(provider)
    patch_benchmark(benchmark)
    print("patched vertex-NavHint DiskANN with online semantic result cache")


if __name__ == "__main__":
    main()
