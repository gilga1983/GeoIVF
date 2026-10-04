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

Optional static PubMed/MedCPT portal routing:
DISKANN_IP_PORTAL_ROUTER_FILE and DISKANN_IP_PORTAL_NPROBE (default 32).
When both portal and Catapult are enabled, the portal replaces the medoid as
the base start and Catapult contributes its learned bucket destinations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import PINNED, once, patch_provider


def patch_provider_medoid(path: Path) -> None:
    s = path.read_text()

    # The generic DiskANN scratch allocator normally adds the number of start
    # points to the candidate-queue capacity. That behavior is appropriate for
    # frozen graph points, but it would turn CatapultDB's 40-entry bucket into
    # an unintended L+40 search. Algorithm 1 in the paper instead evaluates the
    # starting points and trims the candidate set back to k. Keep the ordinary
    # single-medoid scratch allowance while still scoring every custom start.
    s = once(
        s,
        """        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    fn expand_beam<Itr, P, F>(
""",
        """        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
        // Preserve the released single-medoid search budget even when the
        // Catapult layer supplies many candidate entry IDs.
        Ok(1)
    }

    fn expand_beam<Itr, P, F>(
""",
        "fixed start-point scratch accounting",
    )

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

    s = once(
        s,
        """        let recall = if let Some(var_gt) = &gt_context.gt_ids_variable_length {
""",
        """        let recall = if std::env::var_os("DISKANN_SKIP_RECALL").is_some() {
            -1.0
        } else if let Some(var_gt) = &gt_context.gt_ids_variable_length {
""",
        "throughput-only recall bypass",
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

#[derive(Debug)]
struct IpPortalRouter {
    nlist: usize,
    dim: usize,
    centers: Vec<f32>,
    portal_ids: Vec<u32>,
    portal_vectors: Vec<f32>,
}

impl IpPortalRouter {
    fn load(path: &std::path::Path, expected_dim: usize) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GIPIP001";
        if raw.len() < 16 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid inner-product portal-router header");
        }
        let nlist = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let dim = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        if nlist == 0 || dim == 0 || dim != expected_dim {
            anyhow::bail!(
                "portal-router shape mismatch: {}x{}, query dim {}",
                nlist,
                dim,
                expected_dim
            );
        }

        let vector_floats = nlist
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("portal-router shape overflow"))?;
        let vector_bytes = vector_floats
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("portal-router byte-size overflow"))?;
        let id_bytes = nlist
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("portal-router byte-size overflow"))?;
        let expected = 16usize
            .checked_add(vector_bytes)
            .and_then(|n| n.checked_add(id_bytes))
            .and_then(|n| n.checked_add(vector_bytes))
            .ok_or_else(|| anyhow::anyhow!("portal-router byte-size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "portal-router byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }

        let mut offset = 16usize;
        let read_f32_vec =
            |raw: &[u8], offset: &mut usize, count: usize| -> anyhow::Result<Vec<f32>> {
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let x = f32::from_le_bytes(raw[*offset..*offset + 4].try_into()?);
                    *offset += 4;
                    if !x.is_finite() {
                        anyhow::bail!("portal-router contains a nonfinite coordinate");
                    }
                    out.push(x);
                }
                Ok(out)
            };

        let centers = read_f32_vec(&raw, &mut offset, vector_floats)?;
        let mut portal_ids = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            portal_ids.push(u32::from_le_bytes(raw[offset..offset + 4].try_into()?));
            offset += 4;
        }
        let portal_vectors = read_f32_vec(&raw, &mut offset, vector_floats)?;
        if offset != raw.len() {
            anyhow::bail!("portal-router parse did not consume all bytes");
        }

        Ok(Self {
            nlist,
            dim,
            centers,
            portal_ids,
            portal_vectors,
        })
    }

    #[inline]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
    }

    fn route<T: VectorRepr>(&self, query: &[T], nprobe: usize) -> anyhow::Result<u32> {
        let q = T::as_f32(query)
            .map_err(|e| anyhow::anyhow!("query conversion for portal routing failed: {:?}", e))?;
        let q: &[f32] = &q;
        if q.len() != self.dim {
            anyhow::bail!(
                "portal-router query dimension mismatch: got {}, expected {}",
                q.len(),
                self.dim
            );
        }
        let k = nprobe.min(self.nlist);
        if k == 0 || k > 32 {
            anyhow::bail!("portal-router nprobe must be in 1..=32");
        }

        // Keep the k highest-inner-product coarse cells.
        let mut best_storage = [(f32::NEG_INFINITY, usize::MAX); 32];
        let best = &mut best_storage[..k];
        for cell in 0..self.nlist {
            let base = cell * self.dim;
            let score = Self::dot(q, &self.centers[base..base + self.dim]);
            if score <= best[k - 1].0 {
                continue;
            }
            let mut pos = k - 1;
            while pos > 0 && score > best[pos - 1].0 {
                best[pos] = best[pos - 1];
                pos -= 1;
            }
            best[pos] = (score, cell);
        }

        // Among portals from shortlisted cells, choose maximum query inner product.
        let mut winner = best[0].1;
        let mut winner_score = f32::NEG_INFINITY;
        for &(_, cell) in best.iter() {
            let base = cell * self.dim;
            let score = Self::dot(q, &self.portal_vectors[base..base + self.dim]);
            if score > winner_score {
                winner_score = score;
                winner = cell;
            }
        }
        Ok(self.portal_ids[winner])
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

    fn starting_points<T: VectorRepr>(
        &self,
        query: &[T],
        base_start: u32,
    ) -> anyhow::Result<(usize, Vec<u32>)> {
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
        starts.push(base_start);
        for &id in guard.iter() {
            if id != base_start && !starts.contains(&id) {
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

    let portal_router = match std::env::var_os("DISKANN_IP_PORTAL_ROUTER_FILE") {
        Some(path) => Some(IpPortalRouter::load(
            std::path::Path::new(&path),
            queries.ncols(),
        )?),
        None => None,
    };
    let portal_nprobe = match portal_router.as_ref() {
        Some(_) => std::env::var("DISKANN_IP_PORTAL_NPROBE")
            .unwrap_or_else(|_| "32".to_string())
            .parse::<usize>()?,
        None => 0,
    };
    if portal_router.is_some() && !(1..=32).contains(&portal_nprobe) {
        anyhow::bail!("DISKANN_IP_PORTAL_NPROBE must be in 1..=32");
    }

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
    new_call = r'''                let portal_seed = match portal_router.as_ref() {
                    Some(router) => match router.route::<T>(q, portal_nprobe) {
                        Ok(id) => Some(id),
                        Err(e) => {
                            eprintln!("Portal routing failed for query: {:?}", e);
                            *rc = 0;
                            id_chunk.fill(0);
                            dist_chunk.fill(0.0);
                            has_any_search_failed.store(true, Ordering::Release);
                            return;
                        }
                    },
                    None => None,
                };

                // Four clean arms emerge from the two optional mechanisms:
                // medoid; medoid+Catapult; portal; portal+Catapult.
                let routed: Option<(Option<usize>, Vec<u32>)> = match catapult.as_ref() {
                    Some(c) => {
                        let base_start = portal_seed.unwrap_or(c.medoid);
                        match c.starting_points::<T>(q, base_start) {
                            Ok((bucket, starts)) => Some((Some(bucket), starts)),
                            Err(e) => {
                                eprintln!("Catapult routing failed for query: {:?}", e);
                                *rc = 0;
                                id_chunk.fill(0);
                                dist_chunk.fill(0.0);
                                has_any_search_failed.store(true, Ordering::Release);
                                return;
                            }
                        }
                    }
                    None => portal_seed.map(|id| (None, vec![id])),
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

                        if let (Some(c), Some((Some(bucket), _))) = (catapult.as_ref(), routed.as_ref()) {
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

    s = once(
        s,
        """fn prepare_ground_truth_context(
    has_vector_filters: bool,
    groundtruth: &InputFile,
    recall_at: u32,
    storage: &impl StorageReadProvider,
) -> anyhow::Result<GroundTruthContext> {
    let path = groundtruth.to_string_lossy().into_owned();

""",
        """fn prepare_ground_truth_context(
    has_vector_filters: bool,
    groundtruth: &InputFile,
    recall_at: u32,
    storage: &impl StorageReadProvider,
) -> anyhow::Result<GroundTruthContext> {
    if std::env::var_os("DISKANN_SKIP_RECALL").is_some() {
        return Ok(GroundTruthContext {
            gt_ids: None,
            gt_ids_variable_length: None,
            gt_dists: None,
            gt_dim: 0,
            recall_at,
        });
    }

    let path = groundtruth.to_string_lossy().into_owned();

""",
        "throughput-only groundtruth bypass",
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
