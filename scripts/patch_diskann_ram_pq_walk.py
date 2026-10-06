#!/usr/bin/env python3
"""Patch optimized Hint-IVF DiskANN with an optional compact RAM PQ graph walk.

Apply after patch_diskann_hint_ivf_packed_direct.py. The RAM graph contains only
fixed-degree vertex IDs. Search begins from the exact Hint-IVF winner, traverses
the RAM graph using DiskANN's already-prepared PQ query LUT and resident PQ
codes, and emits the best one or more discovered IDs as ordinary graph starts.

Environment:
  DISKANN_RAM_PQ_GRAPH   GIPQG001 fixed-degree adjacency file
  DISKANN_RAM_PQ_BUDGET number of RAM vertices to expand (default 16)
  DISKANN_RAM_PQ_SEEDS  number of best starts to emit (default 2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_start_points import once


def patch_provider(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    sync::{
        atomic::{AtomicU64, AtomicUsize},
        Arc,
    },
""",
        """    sync::{
        atomic::{AtomicU64, AtomicUsize},
        Arc, OnceLock,
    },
""",
        "OnceLock import",
    )

    marker = """#[derive(Clone, Copy)]
struct HintIvfSearch<'a> {
"""
    helper = r'''struct RamPqGraph {
    num_points: usize,
    degree: usize,
    budget: usize,
    seed_count: usize,
    neighbors: Box<[u32]>,
}

static RAM_PQ_GRAPH: OnceLock<Result<Option<RamPqGraph>, String>> = OnceLock::new();

fn configured_ram_pq_graph(expected_points: usize) -> ANNResult<Option<&'static RamPqGraph>> {
    let loaded = RAM_PQ_GRAPH.get_or_init(|| {
        let Some(path) = std::env::var_os("DISKANN_RAM_PQ_GRAPH") else {
            return Ok(None);
        };
        let budget = std::env::var("DISKANN_RAM_PQ_BUDGET")
            .ok()
            .map(|x| x.parse::<usize>())
            .transpose()
            .map_err(|e| format!("invalid DISKANN_RAM_PQ_BUDGET: {e}"))?
            .unwrap_or(16);
        let seed_count = std::env::var("DISKANN_RAM_PQ_SEEDS")
            .ok()
            .map(|x| x.parse::<usize>())
            .transpose()
            .map_err(|e| format!("invalid DISKANN_RAM_PQ_SEEDS: {e}"))?
            .unwrap_or(2);
        if budget == 0 || seed_count == 0 || seed_count > 16 {
            return Err("RAM PQ budget must be positive and seeds must be in 1..=16".to_string());
        }

        let raw = std::fs::read(&path)
            .map_err(|e| format!("failed to read RAM PQ graph {:?}: {e}", path))?;
        const MAGIC: &[u8; 8] = b"GIPQG001";
        if raw.len() < 16 || &raw[..8] != MAGIC {
            return Err("invalid RAM PQ graph header".to_string());
        }
        let n = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let degree = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        if n == 0 || degree == 0 {
            return Err("empty RAM PQ graph".to_string());
        }
        let entries = n
            .checked_mul(degree)
            .ok_or_else(|| "RAM PQ graph size overflow".to_string())?;
        let expected = 16usize
            .checked_add(entries.checked_mul(4).ok_or_else(|| "RAM PQ byte overflow".to_string())?)
            .ok_or_else(|| "RAM PQ byte overflow".to_string())?;
        if raw.len() != expected {
            return Err(format!(
                "RAM PQ graph byte length mismatch: got {}, expected {}",
                raw.len(), expected
            ));
        }
        let mut neighbors = Vec::with_capacity(entries);
        for chunk in raw[16..].chunks_exact(4) {
            let id = u32::from_le_bytes(chunk.try_into().unwrap());
            if id != u32::MAX && id as usize >= n {
                return Err("RAM PQ graph contains out-of-range ID".to_string());
            }
            neighbors.push(id);
        }
        eprintln!(
            "RAM PQ graph loaded: points={} degree={} bytes={} budget={} seeds={}",
            n,
            degree,
            neighbors.len() * 4,
            budget,
            seed_count
        );
        Ok(Some(RamPqGraph {
            num_points: n,
            degree,
            budget,
            seed_count,
            neighbors: neighbors.into_boxed_slice(),
        }))
    });

    match loaded {
        Err(message) => Err(diskann_error!(ErrorKind::IndexError, "{message}")),
        Ok(None) => Ok(None),
        Ok(Some(graph)) => {
            if graph.num_points != expected_points {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "RAM PQ graph point count mismatch",
                ));
            }
            Ok(Some(graph))
        }
    }
}

'''
    s = once(s, marker, helper + marker, "RAM PQ graph loader")

    # Add a RAM-only best-first walk helper to the existing DiskAccessor impl
    # that already contains pq_distances_packed.
    anchor = """impl<Data, VP> HasId for DiskAccessor<'_, Data, VP>
"""
    helper2 = r'''impl<Data, VP> DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    fn emit_hint_or_ram_starts<F>(
        &mut self,
        winner: (f32, u32),
        base_routing_comparisons: usize,
        f: &mut F,
    ) -> ANNResult<()>
    where
        F: FnMut(u32, f32),
    {
        let Some(graph) = configured_ram_pq_graph(self.provider.num_points)? else {
            self.io_tracker.routing_comparisons.fetch_add(
                base_routing_comparisons.saturating_sub(1),
                std::sync::atomic::Ordering::Relaxed,
            );
            f(winner.1, winner.0);
            return Ok(());
        };

        let reserve = 1usize.saturating_add(graph.budget.saturating_mul(graph.degree));
        let mut discovered: Vec<(f32, u32)> = Vec::with_capacity(reserve);
        discovered.push(winner);
        let mut expanded: Vec<u32> = Vec::with_capacity(graph.budget);
        let mut routing_comparisons = base_routing_comparisons;

        for _ in 0..graph.budget {
            let mut best: Option<(f32, u32)> = None;
            for &(distance, id) in &discovered {
                if expanded.contains(&id) {
                    continue;
                }
                let better = best.is_none_or(|current| {
                    distance
                        .total_cmp(&current.0)
                        .then_with(|| id.cmp(&current.1))
                        .is_lt()
                });
                if better {
                    best = Some((distance, id));
                }
            }
            let Some((_, vertex)) = best else {
                break;
            };
            expanded.push(vertex);

            let lo = (vertex as usize)
                .checked_mul(graph.degree)
                .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "RAM PQ offset overflow"))?;
            let hi = lo + graph.degree;

            let mut candidates = std::mem::take(&mut self.scratch.hint_ids_scratch);
            candidates.clear();
            for &id in &graph.neighbors[lo..hi] {
                if id == u32::MAX
                    || discovered.iter().any(|entry| entry.1 == id)
                    || candidates.contains(&id)
                {
                    continue;
                }
                candidates.push(id);
            }

            routing_comparisons = routing_comparisons
                .checked_add(candidates.len())
                .ok_or_else(|| {
                    diskann_error!(ErrorKind::IndexError, "RAM PQ comparison overflow")
                })?;

            let score_result = self.pq_distances(&candidates, |distance, id| {
                discovered.push((distance, id));
            });
            candidates.clear();
            self.scratch.hint_ids_scratch = candidates;
            score_result?;
        }

        discovered.sort_by(|a, b| {
            a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
        });
        let emit = graph.seed_count.min(discovered.len());
        if emit == 0 {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "RAM PQ walk produced no start points",
            ));
        }

        self.io_tracker.routing_comparisons.fetch_add(
            routing_comparisons.saturating_sub(emit),
            std::sync::atomic::Ordering::Relaxed,
        );
        for &(distance, id) in discovered.iter().take(emit) {
            f(id, distance);
        }
        Ok(())
    }
}

'''
    s = once(s, anchor, helper2 + anchor, "RAM PQ walk helper")

    old = """            // The graph search counts the emitted start once. Record the other
            // routing PQ scores without double-counting the winner.
            self.io_tracker
                .routing_comparisons
                .fetch_add(routing_cmps.saturating_sub(1), std::sync::atomic::Ordering::Relaxed);
            f(winner.1, winner.0);
            return Ok(());
"""
    new = """            // Optionally advance farther in the compact RAM graph using the
            // already-prepared PQ query LUT. When disabled, this exactly emits
            // the original Hint-IVF winner.
            self.emit_hint_or_ram_starts(winner, routing_cmps, &mut f)?;
            return Ok(());
"""
    s = once(s, old, new, "integrate RAM PQ walk after Hint-IVF")

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    provider = args.diskann.resolve() / "diskann-disk/src/search/provider/disk_provider.rs"
    if not provider.is_file():
        raise SystemExit("unexpected DiskANN checkout")
    patch_provider(provider)
    print("patched optimized Hint-IVF with integrated RAM PQ graph walk")


if __name__ == "__main__":
    main()
