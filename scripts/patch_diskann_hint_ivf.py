#!/usr/bin/env python3
"""Patch pinned DiskANN with an ID-only IVF over learned navigation hints.

The IVF file stores only representative database IDs, CSR offsets, and learned
hint IDs. Runtime routing reuses DiskANN's resident PQ codes:
  1) score all representative IDs,
  2) probe the best nprobe buckets,
  3) score only hint IDs in those buckets,
  4) pass the best max_starts IDs to ordinary DiskANN graph search.

The routing work is executed inside the benchmark's timed query path. No full
vectors, PQ-code copies, adjacency lists, or auxiliary graph are stored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import PINNED, once, patch_provider
from patch_diskann_paper_catapult import patch_provider_medoid
from patch_diskann_waypoint_cache import patch_provider_waypoint
from patch_diskann_global_starts import patch_benchmark_global


def patch_provider_hint_ivf(path: Path) -> None:
    s = path.read_text()

    marker = """    /// Perform a search on the disk index and return each result with its native indexed vector.
"""
    method = r'''    /// Select graph starts from an ID-only IVF using the index's resident PQ codes.
    ///
    /// This performs no disk I/O. The returned IDs are then passed to the
    /// ordinary caller-supplied-start path, so the graph-search L budget is unchanged.
    pub fn select_hint_ivf_starts(
        &self,
        query: &[Data::VectorDataType],
        medoid_ids: &[u32],
        offsets: &[u32],
        hint_ids: &[u32],
        nprobe: usize,
        max_starts: usize,
    ) -> ANNResult<(Vec<u32>, u32, u128, u128)> {
        let selector_timer = Instant::now();
        if medoid_ids.is_empty()
            || offsets.len() != medoid_ids.len() + 1
            || offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(hint_ids.len() as u32)
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid hint-IVF shape",
            ));
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "hint-IVF offsets are not monotone",
            ));
        }
        if nprobe == 0 || nprobe > medoid_ids.len() || max_starts == 0 {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid hint-IVF search parameters",
            ));
        }

        let num_points = self.index.provider().num_points;
        if medoid_ids
            .iter()
            .chain(hint_ids.iter())
            .any(|id| (*id as usize) >= num_points)
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "hint-IVF ID outside graph range",
            ));
        }

        // A fresh accessor prepares exactly the same PQ lookup table used by
        // ordinary graph traversal. No SSD access occurs in this selector.
        let io_tracker = IOTracker::default();
        let strategy = self.search_strategy(
            &io_tracker,
            PostprocessStrategy::AcceptAll,
            false,
        );
        let mut accessor =
            DiskAccessor::new(self.index.provider(), query, &strategy)?;
        let selector_preprocess_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;

        let mut coarse = Vec::<(f32, usize, u32)>::with_capacity(medoid_ids.len());
        let mut coarse_pos = 0usize;
        accessor.pq_distances(medoid_ids, |distance, id| {
            coarse.push((distance, coarse_pos, id));
            coarse_pos += 1;
        })?;
        coarse.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        coarse.truncate(nprobe);

        // Include selected representatives themselves as valid starts, then
        // score only the children of the selected buckets.
        let mut fine = Vec::<(f32, u32)>::new();
        let mut children = Vec::<u32>::new();
        for &(distance, cell, medoid_id) in &coarse {
            fine.push((distance, medoid_id));
            let lo = offsets[cell] as usize;
            let hi = offsets[cell + 1] as usize;
            children.extend_from_slice(&hint_ids[lo..hi]);
        }

        accessor.pq_distances(&children, |distance, id| {
            fine.push((distance, id));
        })?;
        fine.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let mut starts = Vec::with_capacity(max_starts.min(fine.len()));
        for (_, id) in fine {
            if !starts.contains(&id) {
                starts.push(id);
                if starts.len() == max_starts {
                    break;
                }
            }
        }
        if starts.is_empty() {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "hint-IVF selected no start points",
            ));
        }
        let selector_comparisons = medoid_ids
            .len()
            .checked_add(children.len())
            .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "hint-IVF comparison overflow"))?
            as u32;
        let selector_total_us = selector_timer.elapsed().as_micros();
        Ok((
            starts,
            selector_comparisons,
            selector_total_us,
            selector_preprocess_us,
        ))
    }

'''
    s = once(s, marker, method + marker, "hint-IVF selector method")
    path.write_text(s)


def patch_benchmark_hint_ivf(path: Path) -> None:
    s = path.read_text()

    old_load = """    let global_start_ids = match std::env::var_os("DISKANN_GLOBAL_START_IDS_FILE") {
        Some(path) => load_global_start_ids(std::path::Path::new(&path))?,
        None => Vec::<u32>::new(),
    };

    // Load the vector filters
"""
    new_load = """    let global_start_ids = match std::env::var_os("DISKANN_GLOBAL_START_IDS_FILE") {
        Some(path) => load_global_start_ids(std::path::Path::new(&path))?,
        None => Vec::<u32>::new(),
    };
    let hint_ivf = match std::env::var_os("DISKANN_HINT_IVF_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let hint_ivf_nprobe = std::env::var("DISKANN_HINT_IVF_NPROBE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let hint_ivf_max_starts = std::env::var("DISKANN_HINT_IVF_MAX_STARTS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if let Some(index) = hint_ivf.as_ref() {
        if hint_ivf_nprobe == 0 || hint_ivf_nprobe > index.medoid_ids.len() {
            anyhow::bail!("DISKANN_HINT_IVF_NPROBE outside valid range");
        }
        if hint_ivf_max_starts == 0 {
            anyhow::bail!("DISKANN_HINT_IVF_MAX_STARTS must be positive");
        }
    }

    // Load the vector filters
"""
    s = once(s, old_load, new_load, "load hint-IVF")

    old_call = """                let active_seeds: &[u32] = if !global_start_ids.is_empty() {
                    global_start_ids.as_slice()
                } else {
                    seeds.as_slice()
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
    new_call = """                let mut hint_route_comparisons = 0u32;
                let mut hint_route_total_us = 0u128;
                let mut hint_route_preprocess_us = 0u128;
                let hint_seeds = match hint_ivf.as_ref() {
                    Some(index) => match searcher.select_hint_ivf_starts(
                        q,
                        &index.medoid_ids,
                        &index.offsets,
                        &index.hint_ids,
                        hint_ivf_nprobe,
                        hint_ivf_max_starts,
                    ) {
                        Ok((ids, comparisons, total_us, preprocess_us)) => {
                            hint_route_comparisons = comparisons;
                            hint_route_total_us = total_us;
                            hint_route_preprocess_us = preprocess_us;
                            ids
                        }
                        Err(e) => {
                            eprintln!("Hint-IVF routing failed for query: {:?}", e);
                            *rc = 0;
                            id_chunk.fill(0);
                            dist_chunk.fill(0.0);
                            has_any_search_failed.store(
                                true,
                                std::sync::atomic::Ordering::Release,
                            );
                            return;
                        }
                    },
                    None => Vec::<u32>::new(),
                };
                let active_seeds: &[u32] = if !hint_seeds.is_empty() {
                    hint_seeds.as_slice()
                } else if !global_start_ids.is_empty() {
                    global_start_ids.as_slice()
                } else {
                    seeds.as_slice()
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
    s = once(s, old_call, new_call, "hint-IVF-aware search call")

    marker = """fn load_global_start_ids(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
"""
    helper = r'''struct HintIvfIndex {
    medoid_ids: Vec<u32>,
    offsets: Vec<u32>,
    hint_ids: Vec<u32>,
}

impl HintIvfIndex {
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GHIVF001";
        if raw.len() < 24 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid hint-IVF header");
        }
        let nlist = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let child_count = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let total_landmarks = u32::from_le_bytes(raw[16..20].try_into()?) as usize;
        let reserved = u32::from_le_bytes(raw[20..24].try_into()?);
        if nlist == 0 || child_count + nlist != total_landmarks || reserved != 0 {
            anyhow::bail!("invalid hint-IVF shape");
        }
        let expected = 24usize
            .checked_add(nlist.checked_mul(4).ok_or_else(|| anyhow::anyhow!("hint-IVF overflow"))?)
            .and_then(|x| x.checked_add((nlist + 1).checked_mul(4)?))
            .and_then(|x| x.checked_add(child_count.checked_mul(4)?))
            .ok_or_else(|| anyhow::anyhow!("hint-IVF size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "hint-IVF byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }

        let mut off = 24usize;
        let mut medoid_ids = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            medoid_ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        let mut offsets = Vec::with_capacity(nlist + 1);
        for _ in 0..=nlist {
            offsets.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(child_count as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            anyhow::bail!("invalid hint-IVF offsets");
        }
        let mut hint_ids = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            hint_ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }

        let mut unique = HashSet::with_capacity(total_landmarks);
        for id in medoid_ids.iter().chain(hint_ids.iter()) {
            if !unique.insert(*id) {
                anyhow::bail!("duplicate ID in hint-IVF");
            }
        }
        Ok(Self {
            medoid_ids,
            offsets,
            hint_ids,
        })
    }
}

'''
    s = once(s, marker, helper + marker, "hint-IVF parser")
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
    patch_provider_waypoint(provider)
    patch_benchmark_global(benchmark)
    patch_provider_hint_ivf(provider)
    patch_benchmark_hint_ivf(benchmark)
    print(f"patched DiskANN {PINNED} with ID-only hint IVF")


if __name__ == "__main__":
    main()
