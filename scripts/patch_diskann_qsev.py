#!/usr/bin/env python3
"""Patch pinned DiskANN with an in-process DiskANN++-style QSEV selector.

The deployed QSEV pool contains full-precision graph entry vectors and IDs.
Each query linearly scans the pool under inner product, chooses one graph entry,
and then executes the ordinary DiskANN graph search from that entry. QSEV
selection is inside the timed query path and QPS measurement.

The patch also preserves the existing opt-in per-query start-point hook and
throughput-only recall bypass used by the controlled SSD harness.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import PINNED, once, patch_benchmark, patch_provider


def patch_qsev_benchmark(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        "use diskann::utils::VectorRepr;\n",
        """use diskann::utils::VectorRepr;
use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};
""",
        "QSEV SIMD import",
    )

    marker = "pub(super) fn search_disk_index<T, StorageType>(\n"
    helper = r"""#[derive(Debug)]
struct QsevPool {
    dim: usize,
    ids: Vec<u32>,
    vectors: Vec<f32>,
}

impl QsevPool {
    fn load(path: &std::path::Path, expected_dim: usize) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GQSEV001";
        if raw.len() < 16 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid QSEV pool header");
        }
        let count = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let dim = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        if count == 0 || dim == 0 || dim != expected_dim {
            anyhow::bail!(
                "QSEV pool shape mismatch: {}x{}, query dim {}",
                count,
                dim,
                expected_dim
            );
        }
        let ids_bytes = count
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("QSEV size overflow"))?;
        let vector_floats = count
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("QSEV size overflow"))?;
        let vector_bytes = vector_floats
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("QSEV size overflow"))?;
        let expected = 16usize
            .checked_add(ids_bytes)
            .and_then(|x| x.checked_add(vector_bytes))
            .ok_or_else(|| anyhow::anyhow!("QSEV size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "QSEV pool byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }

        let mut offset = 16usize;
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(u32::from_le_bytes(raw[offset..offset + 4].try_into()?));
            offset += 4;
        }
        let mut vectors = Vec::with_capacity(vector_floats);
        for _ in 0..vector_floats {
            let x = f32::from_le_bytes(raw[offset..offset + 4].try_into()?);
            offset += 4;
            if !x.is_finite() {
                anyhow::bail!("QSEV pool contains nonfinite coordinate");
            }
            vectors.push(x);
        }
        Ok(Self { dim, ids, vectors })
    }

    #[inline]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        <InnerProduct as PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>>>::evaluate(a, b)
            .into_inner()
    }

    fn route<T: VectorRepr>(&self, query: &[T]) -> anyhow::Result<u32> {
        let q = T::as_f32(query)
            .map_err(|e| anyhow::anyhow!("QSEV query conversion failed: {:?}", e))?;
        let q: &[f32] = &q;
        if q.len() != self.dim {
            anyhow::bail!(
                "QSEV query dimension mismatch: got {}, expected {}",
                q.len(),
                self.dim
            );
        }

        let mut best_id = self.ids[0];
        let mut best_score = f32::NEG_INFINITY;
        for (i, &id) in self.ids.iter().enumerate() {
            let base = i * self.dim;
            let score = Self::dot(q, &self.vectors[base..base + self.dim]);
            if score > best_score {
                best_score = score;
                best_id = id;
            }
        }
        Ok(best_id)
    }
}

"""
    s = once(s, marker, helper + marker, "QSEV pool implementation")

    s = once(
        s,
        """    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };

    // Load the vector filters
""",
        """    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };
    let qsev_pool = match std::env::var_os("DISKANN_QSEV_FILE") {
        Some(path) => Some(QsevPool::load(std::path::Path::new(&path), queries.ncols())?),
        None => None,
    };

    // Load the vector filters
""",
        "load QSEV pool",
    )

    old_call = """                let result = if seeds.is_empty() {
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
    new_call = r"""                let query_timer = Instant::now();
                let qsev_id = match qsev_pool.as_ref() {
                    Some(pool) => match pool.route::<T>(q) {
                        Ok(id) => Some(id),
                        Err(e) => {
                            eprintln!("QSEV routing failed for query: {:?}", e);
                            *rc = 0;
                            id_chunk.fill(0);
                            dist_chunk.fill(0.0);
                            has_any_search_failed.store(true, std::sync::atomic::Ordering::Release);
                            return;
                        }
                    },
                    None => None,
                };
                let qsev_one = qsev_id.map(|id| [id]);
                let active_seeds: &[u32] = if !seeds.is_empty() {
                    seeds.as_slice()
                } else if let Some(ref one) = qsev_one {
                    one.as_slice()
                } else {
                    &[]
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
    s = once(s, old_call, new_call, "QSEV-aware search call")

    s = once(
        s,
        """                    Ok(search_result) => {
                        *stats = search_result.stats.query_statistics;
                        let base_count = (search_result.stats.result_count as usize)
""",
        """                    Ok(search_result) => {
                        *stats = search_result.stats.query_statistics;
                        stats.total_execution_time_us = query_timer.elapsed().as_micros();
                        let base_count = (search_result.stats.result_count as usize)
""",
        "include QSEV routing in query latency",
    )

    s = once(
        s,
        """        let recall = if let Some(var_gt) = &gt_context.gt_ids_variable_length {
""",
        """        let recall = if std::env::var_os("DISKANN_SKIP_RECALL").is_some() {
            -1.0
        } else if let Some(var_gt) = &gt_context.gt_ids_variable_length {
""",
        "throughput-only recall sentinel",
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
    patch_benchmark(benchmark)
    patch_qsev_benchmark(benchmark)
    print(f"patched DiskANN {PINNED} with in-process QSEV entry selection")


if __name__ == "__main__":
    main()
