#!/usr/bin/env python3
"""Patch pinned DiskANN for one global PQ-scored start-ID pool.

The start pool is shared by all queries and contains only database IDs. DiskANN
scores the IDs using its existing resident PQ representation during ordinary
search initialization. The candidate queue budget stays at the released
single-medoid allowance, so thousands of candidate starts do not widen L.

Enable with DISKANN_GLOBAL_START_IDS_FILE. File format:
  magic[8] = GIDST001
  count:u32
  reserved:u32
  ids[count]:u32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_start_points import (
    PINNED,
    once,
    patch_provider,
    patch_benchmark as patch_start_benchmark,
)
from patch_diskann_paper_catapult import patch_provider_medoid
from patch_diskann_waypoint_cache import patch_provider_waypoint


def patch_benchmark_global(path: Path) -> None:
    # First install the ordinary per-query start-point hook.
    patch_start_benchmark(path)
    s = path.read_text()

    old = """    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };

    // Load the vector filters
"""
    new = """    let seed_rows = match std::env::var_os("DISKANN_START_POINTS_FILE") {
        Some(path) => load_start_points(std::path::Path::new(&path), num_queries)?,
        None => vec![Vec::<u32>::new(); num_queries],
    };
    let global_start_ids = match std::env::var_os("DISKANN_GLOBAL_START_IDS_FILE") {
        Some(path) => load_global_start_ids(std::path::Path::new(&path))?,
        None => Vec::<u32>::new(),
    };

    // Load the vector filters
"""
    s = once(s, old, new, "load global start IDs")

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
    new_call = """                let active_seeds: &[u32] = if !global_start_ids.is_empty() {
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
    s = once(s, old_call, new_call, "global-start-aware call")

    marker = """fn load_start_points(path: &std::path::Path, expected_rows: usize) -> anyhow::Result<Vec<Vec<u32>>> {
"""
    helper = r'''fn load_global_start_ids(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
    let raw = std::fs::read(path)?;
    const MAGIC: &[u8; 8] = b"GIDST001";
    if raw.len() < 16 || &raw[0..8] != MAGIC {
        anyhow::bail!("invalid global-start header");
    }
    let count = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
    let reserved = u32::from_le_bytes(raw[12..16].try_into()?);
    if count == 0 || reserved != 0 {
        anyhow::bail!("invalid global-start shape");
    }
    let expected = 16usize
        .checked_add(
            count
                .checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("global-start size overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("global-start size overflow"))?;
    if raw.len() != expected {
        anyhow::bail!(
            "global-start byte length mismatch: got {}, expected {}",
            raw.len(),
            expected
        );
    }
    let mut ids = Vec::with_capacity(count);
    let mut unique = HashSet::with_capacity(count);
    let mut off = 16usize;
    for _ in 0..count {
        let id = u32::from_le_bytes(raw[off..off + 4].try_into()?);
        off += 4;
        if !unique.insert(id) {
            anyhow::bail!("duplicate global start ID");
        }
        ids.push(id);
    }
    Ok(ids)
}

'''
    s = once(s, marker, helper + marker, "global start parser")
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
    print(f"patched DiskANN {PINNED} for global PQ-scored start-ID pools")


if __name__ == "__main__":
    main()
