#!/usr/bin/env python3
"""Patch pinned DiskANN3 for in-process centroid/portal routing.

This first applies the proven query-specific start-point carrier patch, then adds
an optional tiny portal router inside the DiskANN benchmark process. Routing is
performed per query under the same wall-clock timer as graph traversal and
reranking, so reported latency/QPS include routing overhead.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import PINNED, once, patch_provider, patch_benchmark


def patch_integrated_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use diskann::utils::VectorRepr;
""",
        """use diskann::utils::VectorRepr;
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
""",
        "SIMD L2 import",
    )

    s = once(
        s,
        """    pub(super) mean_latency: f64,
    pub(super) p95_latency: MicroSeconds,
""",
        """    pub(super) mean_latency: f64,
    pub(super) mean_route_latency: f64,
    pub(super) p95_latency: MicroSeconds,
""",
        "route latency result field",
    )

    s = once(
        s,
        """        statistics: &[QueryStatistics],
        result_ids: &[u32],
""",
        """        statistics: &[QueryStatistics],
        route_times_us: &[u128],
        result_ids: &[u32],
""",
        "result constructor route input",
    )

    s = once(
        s,
        """            mean_latency: statistics::get_mean_stats(statistics, |s| {
                s.total_execution_time_us as f64
            }),
            p95_latency: MicroSeconds::new(
""",
        """            mean_latency: statistics::get_mean_stats(statistics, |s| {
                s.total_execution_time_us as f64
            }),
            mean_route_latency: if route_times_us.is_empty() {
                0.0
            } else {
                route_times_us.iter().copied().sum::<u128>() as f64 / route_times_us.len() as f64
            },
            p95_latency: MicroSeconds::new(
""",
        "result route latency calculation",
    )

    s = once(
        s,
        """    pub(super) num_nodes_to_cache: Option<usize>,
    pub(super) search_results_per_l: Vec<DiskSearchResult>,
""",
        """    pub(super) num_nodes_to_cache: Option<usize>,
    pub(super) portal_nprobe: Option<usize>,
    pub(super) search_results_per_l: Vec<DiskSearchResult>,
""",
        "stats portal nprobe field",
    )

    seed_load = """    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };

"""
    integrated_load = seed_load + """    // Optional in-process coarse router. The table is loaded once, while each
    // query's routing computation happens inside the timed search closure.
    let portal_router = match std::env::var_os("DISKANN_PORTAL_ROUTER_FILE") {
        Some(path) => Some(PortalRouter::load(
            std::path::Path::new(&path),
            queries.ncols(),
        )?),
        None => None,
    };
    let portal_nprobe = match portal_router.as_ref() {
        Some(_) => {
            let raw = std::env::var("DISKANN_PORTAL_NPROBE")
                .map_err(|_| anyhow::anyhow!("DISKANN_PORTAL_NPROBE is required with a portal router"))?;
            let nprobe: usize = raw.parse()?;
            if nprobe == 0 {
                anyhow::bail!("DISKANN_PORTAL_NPROBE must be positive");
            }
            nprobe
        }
        None => 0,
    };
    if portal_router.is_some() && std::env::var_os("DISKANN_START_POINTS_FILE").is_some() {
        anyhow::bail!("portal routing and precomputed start-point files are mutually exclusive");
    }

"""
    s = once(s, seed_load, integrated_load, "integrated router load")

    s = once(
        s,
        """        let mut statistics_vec: Vec<QueryStatistics> =
            vec![QueryStatistics::default(); num_queries];
        let mut result_counts: Vec<u32> = vec![0; num_queries];
""",
        """        let mut statistics_vec: Vec<QueryStatistics> =
            vec![QueryStatistics::default(); num_queries];
        let mut route_times_us: Vec<u128> = vec![0; num_queries];
        let mut result_counts: Vec<u32> = vec![0; num_queries];
""",
        "per-query route timing storage",
    )

    s = once(
        s,
        """            .zip(statistics_vec.par_iter_mut())
            .zip(result_counts.par_iter_mut())
            .zip(seed_rows.par_iter());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |((((((q, vf), id_chunk), dist_chunk), stats), rc), seeds)| {
""",
        """            .zip(statistics_vec.par_iter_mut())
            .zip(route_times_us.par_iter_mut())
            .zip(result_counts.par_iter_mut())
            .zip(seed_rows.par_iter());

        zipped.for_each_in_pool(
            pool.as_ref(),
            |(((((((q, vf), id_chunk), dist_chunk), stats), route_us), rc), seeds)| {
                let query_timer = Instant::now();

                let routed_seed = if let Some(router) = portal_router.as_ref() {
                    let route_timer = Instant::now();
                    match router.route::<T>(q, portal_nprobe) {
                        Ok(seed) => {
                            *route_us = route_timer.elapsed().as_micros();
                            Some([seed])
                        }
                        Err(e) => {
                            eprintln!("Portal routing failed for query: {:?}", e);
                            *rc = 0;
                            id_chunk.fill(0);
                            dist_chunk.fill(0.0);
                            has_any_search_failed
                                .store(true, std::sync::atomic::Ordering::Release);
                            return;
                        }
                    }
                } else {
                    None
                };
                let effective_seeds: &[u32] = match routed_seed.as_ref() {
                    Some(ids) => ids,
                    None => seeds.as_slice(),
                };
""",
        "timed integrated routing closure",
    )

    s = once(
        s,
        """                let result = if seeds.is_empty() {
                    searcher.search(
""",
        """                let result = if effective_seeds.is_empty() {
                    searcher.search(
""",
        "integrated seed choice branch",
    )

    s = once(
        s,
        """                        seeds,
                        mode,
                    )
""",
        """                        effective_seeds,
                        mode,
                    )
""",
        "integrated seed call",
    )

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

                        // Replace the searcher's internal timer with the complete
                        // in-process request time: routing, graph search, reranking,
                        // and result extraction.
                        stats.total_execution_time_us = query_timer.elapsed().as_micros();
""",
        "true end-to-end query timer",
    )

    s = once(
        s,
        """            &statistics_vec,
            &result_ids,
""",
        """            &statistics_vec,
            &route_times_us,
            &result_ids,
""",
        "result constructor call route timing",
    )

    s = once(
        s,
        """        num_nodes_to_cache: search_params.num_nodes_to_cache,
        search_results_per_l,
""",
        """        num_nodes_to_cache: search_params.num_nodes_to_cache,
        portal_nprobe: portal_router.as_ref().map(|_| portal_nprobe),
        search_results_per_l,
""",
        "stats portal nprobe assignment",
    )

    marker = """// Simplified internal structures to reduce parameter count
"""
    helper = r'''#[derive(Debug)]
struct PortalRouter {
    nlist: usize,
    dim: usize,
    centers: Vec<f32>,
    portal_ids: Vec<u32>,
    portal_vectors: Vec<f32>,
}

impl PortalRouter {
    fn load(path: &std::path::Path, expected_dim: usize) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GIPORT01";
        if raw.len() < 16 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid portal-router header");
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

        let mut offset = 16;
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

        Ok(Self {
            nlist,
            dim,
            centers,
            portal_ids,
            portal_vectors,
        })
    }

    #[inline]
    fn l2(a: &[f32], b: &[f32]) -> f32 {
        SquaredL2::evaluate(a, b)
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
        if k == 0 {
            anyhow::bail!("portal-router nprobe must be positive");
        }
        if k > 32 {
            anyhow::bail!("portal-router nprobe above frozen maximum of 32");
        }

        // Keep the nearest k coarse cells in a fixed stack buffer. The frozen
        // qualification policy never exceeds 32, so there is no per-query heap
        // allocation in the routing path.
        let mut best_storage = [(f32::INFINITY, usize::MAX); 32];
        let best = &mut best_storage[..k];
        for cell in 0..self.nlist {
            let base = cell * self.dim;
            let dist = Self::l2(&q, &self.centers[base..base + self.dim]);
            if dist >= best[k - 1].0 {
                continue;
            }
            let mut pos = k - 1;
            while pos > 0 && dist < best[pos - 1].0 {
                best[pos] = best[pos - 1];
                pos -= 1;
            }
            best[pos] = (dist, cell);
        }

        // One stored portal per coarse cell. Among the shortlisted cells, use
        // the portal vector closest to the query.
        let mut winner = best[0].1;
        let mut winner_dist = f32::INFINITY;
        for &(_, cell) in best.iter() {
            let base = cell * self.dim;
            let dist = Self::l2(&q, &self.portal_vectors[base..base + self.dim]);
            if dist < winner_dist {
                winner_dist = dist;
                winner = cell;
            }
        }
        Ok(self.portal_ids[winner])
    }
}

'''
    s = once(s, marker, helper + marker, "portal router implementation")

    s = once(
        s,
        """            ("Mean Latency", 13),
            ("95% Latency", 13),
""",
        """            ("Mean Latency", 13),
            ("Route (us)", 10),
            ("95% Latency", 13),
""",
        "display route column header",
    )
    s = s.replace("let cols: [(&str, usize); 14] = [", "let cols: [(&str, usize); 15] = [", 1)
    s = once(
        s,
        """            let vals: [String; 14] = [
                format!("{}", r.search_l),
                format!("{}", self.recall_at),
                format!("{:.1}", r.qps),
                fmt_us(r.mean_latency),
                format!("{}", r.p95_latency),
""",
        """            let vals: [String; 15] = [
                format!("{}", r.search_l),
                format!("{}", self.recall_at),
                format!("{:.1}", r.qps),
                fmt_us(r.mean_latency),
                fmt_us(r.mean_route_latency),
                format!("{}", r.p95_latency),
""",
        "display route column value",
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
    print(f"patched {PINNED} for in-process portal routing and end-to-end timing")


if __name__ == "__main__":
    main()
