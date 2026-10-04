#!/usr/bin/env python3
"""Patch pinned DiskANN with static portal + learned waypoint-cache starts.

Applies the current paper-Catapult/static-IP-portal patch, then adds an optional
per-portal-region waypoint table loaded through DISKANN_WAYPOINT_CACHE_FILE.

When enabled, the query starts from:
  static portal + learned waypoint IDs for the portal's winning region.
If Catapult is also enabled, Catapult destinations are unioned with those starts.
DiskANN's scratch accounting remains fixed to the original single-start budget,
so extra starts are scored in PQ space but do not widen L.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_paper_catapult import (
    PINNED,
    patch_benchmark as patch_catapult_benchmark,
    patch_provider_medoid,
)
from patch_diskann_start_points import once, patch_provider


def patch_waypoint_benchmark(path: Path) -> None:
    s = path.read_text()

    # Expose the winning portal region without changing existing route().
    old_sig = """    fn route<T: VectorRepr>(&self, query: &[T], nprobe: usize) -> anyhow::Result<u32> {
"""
    new_sig = """    fn route_with_cell<T: VectorRepr>(
        &self,
        query: &[T],
        nprobe: usize,
    ) -> anyhow::Result<(usize, u32)> {
"""
    s = once(s, old_sig, new_sig, "portal route-with-cell signature")

    old_return = """        Ok(self.portal_ids[winner])
    }
}

struct PaperCatapult {
"""
    new_return = """        Ok((winner, self.portal_ids[winner]))
    }

    fn route<T: VectorRepr>(&self, query: &[T], nprobe: usize) -> anyhow::Result<u32> {
        Ok(self.route_with_cell::<T>(query, nprobe)?.1)
    }
}

struct WaypointCache {
    offsets: Vec<u32>,
    ids: Vec<u32>,
}

impl WaypointCache {
    fn load(path: &std::path::Path, expected_nlist: usize) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        const MAGIC: &[u8; 8] = b"GIWPT001";
        if raw.len() < 16 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid waypoint-cache header");
        }
        let nlist = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let total = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        if nlist != expected_nlist {
            anyhow::bail!(
                "waypoint-cache nlist mismatch: got {}, expected {}",
                nlist,
                expected_nlist
            );
        }
        let offsets_bytes = (nlist + 1)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("waypoint-cache size overflow"))?;
        let ids_bytes = total
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("waypoint-cache size overflow"))?;
        let expected = 16usize
            .checked_add(offsets_bytes)
            .and_then(|x| x.checked_add(ids_bytes))
            .ok_or_else(|| anyhow::anyhow!("waypoint-cache size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "waypoint-cache byte length mismatch: got {}, expected {}",
                raw.len(),
                expected
            );
        }

        let mut offsets = Vec::with_capacity(nlist + 1);
        let mut off = 16usize;
        for _ in 0..=nlist {
            offsets.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(total as u32)
        {
            anyhow::bail!("waypoint-cache offsets do not span the ID array");
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            anyhow::bail!("waypoint-cache offsets are not monotone");
        }

        let mut ids = Vec::with_capacity(total);
        for _ in 0..total {
            ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        Ok(Self { offsets, ids })
    }

    fn for_cell(&self, cell: usize) -> &[u32] {
        let lo = self.offsets[cell] as usize;
        let hi = self.offsets[cell + 1] as usize;
        &self.ids[lo..hi]
    }
}

struct PaperCatapult {
"""
    s = once(s, old_return, new_return, "waypoint cache implementation")

    old_load = """    if portal_router.is_some() && !(1..=32).contains(&portal_nprobe) {
        anyhow::bail!("DISKANN_IP_PORTAL_NPROBE must be in 1..=32");
    }

    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
"""
    new_load = """    if portal_router.is_some() && !(1..=32).contains(&portal_nprobe) {
        anyhow::bail!("DISKANN_IP_PORTAL_NPROBE must be in 1..=32");
    }

    let waypoint_cache = match std::env::var_os("DISKANN_WAYPOINT_CACHE_FILE") {
        Some(path) => {
            let router = portal_router
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!(
                    "waypoint cache requires DISKANN_IP_PORTAL_ROUTER_FILE"
                ))?;
            Some(WaypointCache::load(std::path::Path::new(&path), router.nlist)?)
        }
        None => None,
    };

    let mut search_results_per_l = Vec::with_capacity(search_params.search_list.len());
"""
    s = once(s, old_load, new_load, "load waypoint cache")

    old_route = r'''                let portal_seed = match portal_router.as_ref() {
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
'''
    new_route = r'''                let portal_route = match portal_router.as_ref() {
                    Some(router) => match router.route_with_cell::<T>(q, portal_nprobe) {
                        Ok(route) => Some(route),
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

                let mut local_starts = Vec::<u32>::new();
                if let Some((cell, portal_id)) = portal_route {
                    local_starts.push(portal_id);
                    if let Some(cache) = waypoint_cache.as_ref() {
                        for &id in cache.for_cell(cell) {
                            if !local_starts.contains(&id) {
                                local_starts.push(id);
                            }
                        }
                    }
                }

                // Compose all optional navigation memories without changing L.
                let routed: Option<(Option<usize>, Vec<u32>)> = match catapult.as_ref() {
                    Some(c) => {
                        let base_start = local_starts.first().copied().unwrap_or(c.medoid);
                        match c.starting_points::<T>(q, base_start) {
                            Ok((bucket, mut starts)) => {
                                for id in local_starts.into_iter().skip(1) {
                                    if !starts.contains(&id) {
                                        starts.push(id);
                                    }
                                }
                                Some((Some(bucket), starts))
                            }
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
                    None => {
                        if local_starts.is_empty() {
                            None
                        } else {
                            Some((None, local_starts))
                        }
                    }
                };
'''
    s = once(s, old_route, new_route, "compose waypoint starts")
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
    patch_catapult_benchmark(benchmark)
    patch_waypoint_benchmark(benchmark)
    print(f"patched DiskANN {PINNED} with learned waypoint-cache starts")


if __name__ == "__main__":
    main()
