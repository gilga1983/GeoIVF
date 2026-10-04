#!/usr/bin/env python3
"""Patch pinned Rust DiskANN with the CatapultDB policy described in the paper.

Paper policy:
- random-hyperplane LSH over queries;
- medoid is always a starting point;
- all destinations in the matching bucket are additional starting points;
- successful query inserts ann.best() into that bucket;
- fixed bucket capacity with LRU-style refresh of repeated destinations;
- one reader/writer lock per bucket for query-level concurrency.

Enable with DISKANN_PAPER_CATAPULT=1. Optional environment:
DISKANN_CATAPULT_HASHES (default 8), DISKANN_CATAPULT_CAPACITY (40),
DISKANN_CATAPULT_SEED (0).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import PINNED, once, patch_provider


def patch_provider_medoid(path: Path) -> None:
    s = path.read_text()
    marker = """    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    helper = """    /// Return the saved Vamana medoid used by the unmodified search path.
    pub fn medoid(&self) -> u32 {
        self.index.provider().graph_header.metadata().medoid as u32
    }

"""
    s = once(s, marker, helper + marker, "medoid accessor")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use rayon::prelude::*;
use std::{collections::HashSet, fmt, sync::atomic::AtomicBool, time::Instant};
""",
        """use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{
    collections::{HashSet, VecDeque},
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        RwLock,
    },
    time::Instant,
};
""",
        "paper catapult imports",
    )

    s = once(
        s,
        """    pub(super) mean_hops: f64,
    pub(super) cache_hit_percentage: f64,
    pub(super) recall: f32,
""",
        """    pub(super) mean_hops: f64,
    pub(super) cache_hit_percentage: f64,
    pub(super) catapult_usage_percentage: f64,
    pub(super) mean_catapult_starts: f64,
    pub(super) recall: f32,
""",
        "catapult result fields",
    )

    s = once(
        s,
        """            mean_hops: statistics::get_mean_stats(statistics, |s| s.search_hops as f64),
            cache_hit_percentage,
            recall,
""",
        """            mean_hops: statistics::get_mean_stats(statistics, |s| s.search_hops as f64),
            cache_hit_percentage,
            catapult_usage_percentage: 0.0,
            mean_catapult_starts: 0.0,
            recall,
""",
        "catapult result defaults",
    )

    marker = """pub(super) fn search_disk_index<T, StorageType>(
"""
    helper = r'''#[derive(Debug, Clone, Copy)]
struct PaperCatapultConfig {
    hashes: usize,
    capacity: usize,
    seed: u64,
}

impl PaperCatapultConfig {
    fn from_env() -> anyhow::Result<Option<Self>> {
        let enabled = std::env::var("DISKANN_PAPER_CATAPULT")
            .ok()
            .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "FALSE"))
            .unwrap_or(false);
        if !enabled {
            return Ok(None);
        }

        let hashes = std::env::var("DISKANN_CATAPULT_HASHES")
            .unwrap_or_else(|_| "8".to_string())
            .parse::<usize>()?;
        let capacity = std::env::var("DISKANN_CATAPULT_CAPACITY")
            .unwrap_or_else(|_| "40".to_string())
            .parse::<usize>()?;
        let seed = std::env::var("DISKANN_CATAPULT_SEED")
            .unwrap_or_else(|_| "0".to_string())
            .parse::<u64>()?;

        if hashes == 0 || hashes > 16 {
            anyhow::bail!("DISKANN_CATAPULT_HASHES must be in 1..=16");
        }
        if capacity == 0 {
            anyhow::bail!("DISKANN_CATAPULT_CAPACITY must be positive");
        }

        Ok(Some(Self { hashes, capacity, seed }))
    }
}

struct PaperCatapult {
    dim: usize,
    hashes: usize,
    capacity: usize,
    medoid: u32,
    hyperplanes: Vec<f32>,
    buckets: Vec<RwLock<VecDeque<u32>>>,
    total_queries: AtomicU64,
    queries_with_catapult: AtomicU64,
    total_catapult_starts: AtomicU64,
}

impl PaperCatapult {
    fn new(dim: usize, medoid: u32, cfg: PaperCatapultConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut hyperplanes = Vec::with_capacity(cfg.hashes * dim);
        while hyperplanes.len() < cfg.hashes * dim {
            // Box-Muller transform from the already-pinned rand crate.
            // Random-hyperplane LSH needs isotropic normal directions.
            let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2 = rng.random::<f32>();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            hyperplanes.push(radius * theta.cos());
            if hyperplanes.len() < cfg.hashes * dim {
                hyperplanes.push(radius * theta.sin());
            }
        }
        let bucket_count = 1usize << cfg.hashes;
        let buckets = (0..bucket_count)
            .map(|_| RwLock::new(VecDeque::with_capacity(cfg.capacity)))
            .collect();

        Self {
            dim,
            hashes: cfg.hashes,
            capacity: cfg.capacity,
            medoid,
            hyperplanes,
            buckets,
            total_queries: AtomicU64::new(0),
            queries_with_catapult: AtomicU64::new(0),
            total_catapult_starts: AtomicU64::new(0),
        }
    }

    fn bucket<T: VectorRepr>(&self, query: &[T]) -> anyhow::Result<usize> {
        let q = T::as_f32(query)
            .map_err(|e| anyhow::anyhow!("Catapult query conversion failed: {:?}", e))?;
        let q: &[f32] = &q;
        if q.len() != self.dim {
            anyhow::bail!(
                "Catapult query dimension mismatch: got {}, expected {}",
                q.len(),
                self.dim
            );
        }

        let mut code = 0usize;
        for h in 0..self.hashes {
            let plane = &self.hyperplanes[h * self.dim..(h + 1) * self.dim];
            let dot = q
                .iter()
                .zip(plane.iter())
                .map(|(a, b)| *a * *b)
                .sum::<f32>();
            if dot >= 0.0 {
                code |= 1usize << h;
            }
        }
        Ok(code)
    }

    fn starting_points<T: VectorRepr>(&self, query: &[T]) -> anyhow::Result<(usize, Vec<u32>)> {
        let bucket = self.bucket(query)?;
        let guard = self.buckets[bucket]
            .read()
            .map_err(|_| anyhow::anyhow!("Catapult bucket lock poisoned"))?;

        self.total_queries.fetch_add(1, Ordering::Relaxed);
        if !guard.is_empty() {
            self.queries_with_catapult.fetch_add(1, Ordering::Relaxed);
        }
        self.total_catapult_starts
            .fetch_add(guard.len() as u64, Ordering::Relaxed);

        let mut starts = Vec::with_capacity(guard.len() + 1);
        starts.push(self.medoid);
        for &id in guard.iter() {
            if id != self.medoid && !starts.contains(&id) {
                starts.push(id);
            }
        }
        Ok((bucket, starts))
    }

    fn insert(&self, bucket: usize, destination: u32) -> anyhow::Result<()> {
        if destination == self.medoid {
            return Ok(());
        }

        let mut guard = self.buckets[bucket]
            .write()
            .map_err(|_| anyhow::anyhow!("Catapult bucket lock poisoned"))?;

        // Re-observing a destination refreshes it to MRU; otherwise append it.
        if let Some(pos) = guard.iter().position(|&x| x == destination) {
            guard.remove(pos);
        }
        guard.push_back(destination);
        while guard.len() > self.capacity {
            guard.pop_front();
        }
        Ok(())
    }

    fn metrics(&self) -> (f64, f64) {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            return (0.0, 0.0);
        }
        let used = self.queries_with_catapult.load(Ordering::Relaxed);
        let starts = self.total_catapult_starts.load(Ordering::Relaxed);
        (
            100.0 * used as f64 / total as f64,
            starts as f64 / total as f64,
        )
    }
}

'''
    s = once(s, marker, helper + marker, "paper catapult implementation")

    s = once(
        s,
        """    let pool = create_thread_pool(search_params.num_threads)?;
    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
""",
        """    let pool = create_thread_pool(search_params.num_threads)?;
    let catapult_config = PaperCatapultConfig::from_env()?;
    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
""",
        "catapult config load",
    )

    s = once(
        s,
        """        let mut result_dists: Vec<f32> =
            vec![0.0; (search_params.recall_at as usize) * num_queries];

        let start = Instant::now();
""",
        """        let mut result_dists: Vec<f32> =
            vec![0.0; (search_params.recall_at as usize) * num_queries];

        // Each L/K operating point starts with an empty cache, matching an
        // independent paper experiment. The same LSH seed is reused.
        let catapult = catapult_config.map(|cfg| {
            PaperCatapult::new(queries.ncols(), searcher.medoid(), cfg)
        });

        let start = Instant::now();
""",
        "catapult state per operating point",
    )

    old_call = """                match searcher.search(
                    q,
                    search_params.recall_at,
                    l,
                    Some(search_params.beam_width),
                    mode,
                ) {
"""
    new_call = r'''                let routed = match catapult.as_ref() {
                    Some(c) => match c.starting_points::<T>(q) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("Catapult routing failed for query: {:?}", e);
                            *rc = 0;
                            id_chunk.fill(0);
                            dist_chunk.fill(0.0);
                            has_any_search_failed.store(true, Ordering::Release);
                            return;
                        }
                    },
                    None => None,
                };

                let result = match routed.as_ref() {
                    Some((_, starts)) => searcher.search_with_start_points(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        starts,
                        mode,
                    ),
                    None => searcher.search(
                        q,
                        search_params.recall_at,
                        l,
                        Some(search_params.beam_width),
                        mode,
                    ),
                };

                match result {
'''
    s = once(s, old_call, new_call, "catapult-aware search call")

    s = once(
        s,
        """                        for (i, result_item) in
                            search_result.results.iter().take(base_count).enumerate()
                        {
                            id_chunk[i] = result_item.vertex_id;
                            dist_chunk[i] = result_item.distance;
                        }
""",
        """                        for (i, result_item) in
                            search_result.results.iter().take(base_count).enumerate()
                        {
                            id_chunk[i] = result_item.vertex_id;
                            dist_chunk[i] = result_item.distance;
                        }

                        if let (Some(c), Some((bucket, _))) = (catapult.as_ref(), routed.as_ref()) {
                            if base_count > 0 {
                                if let Err(e) = c.insert(*bucket, id_chunk[0]) {
                                    eprintln!("Catapult update failed: {:?}", e);
                                    has_any_search_failed.store(true, Ordering::Release);
                                }
                            }
                        }
""",
        "catapult update",
    )

    s = once(
        s,
        """        let search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;

        l_span.end();
        search_results_per_l.push(search_result);
""",
        """        let mut search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;

        if let Some(c) = catapult.as_ref() {
            let (usage, starts) = c.metrics();
            search_result.catapult_usage_percentage = usage;
            search_result.mean_catapult_starts = starts;
        }

        l_span.end();
        search_results_per_l.push(search_result);
""",
        "catapult result metrics",
    )

    s = s.replace("let cols: [(&str, usize); 14] = [", "let cols: [(&str, usize); 16] = [", 1)
    s = once(
        s,
        """            ("Cache Hit %", 12),
            ("Recall", 7),
""",
        """            ("Cache Hit %", 12),
            ("Catapult %", 10),
            ("Cat Starts", 10),
            ("Recall", 7),
""",
        "catapult display columns",
    )
    s = once(
        s,
        """            let vals: [String; 14] = [
""",
        """            let vals: [String; 16] = [
""",
        "catapult display values size",
    )
    s = once(
        s,
        """                fmt_pct(r.cache_hit_percentage),
                format!("{:.3}", r.recall),
""",
        """                fmt_pct(r.cache_hit_percentage),
                fmt_pct(r.catapult_usage_percentage),
                format!("{:.2}", r.mean_catapult_starts),
                format!("{:.3}", r.recall),
""",
        "catapult display values",
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
    patch_provider_medoid(provider)
    patch_benchmark(benchmark)
    print(f"patched DiskANN {PINNED} with paper-faithful CatapultDB policy")


if __name__ == "__main__":
    main()
