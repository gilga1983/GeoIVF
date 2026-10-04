#!/usr/bin/env python3
"""Patch pinned DiskANN with waypoint-cache support plus Catapult snapshot load/save.

This enables a fair frozen-cache experiment:
1. run paper Catapult on a training prefix and dump its exact bucket state;
2. load that state on a held-out suffix with updates disabled;
3. compare against frozen endpoint/waypoint caches learned from the same prefix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_diskann_waypoint_cache import (
    patch_provider,
    patch_provider_medoid,
    patch_provider_waypoint,
    patch_catapult_benchmark,
    patch_waypoint_benchmark,
)
from patch_diskann_start_points import PINNED, once


def patch_snapshot(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    capacity: usize,
    medoid: u32,
""",
        """    capacity: usize,
    seed: u64,
    medoid: u32,
""",
        "catapult snapshot seed field",
    )

    s = once(
        s,
        """            capacity: cfg.capacity,
            medoid,
""",
        """            capacity: cfg.capacity,
            seed: cfg.seed,
            medoid,
""",
        "catapult snapshot seed init",
    )

    marker = """    fn metrics(&self) -> (f64, f64) {
"""
    helper = r'''    fn save_snapshot(&self, path: &std::path::Path) -> anyhow::Result<()> {
        const MAGIC: &[u8; 8] = b"GICAT001";
        let bucket_count = self.buckets.len();
        let mut flat = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(bucket_count + 1);
        offsets.push(0);
        for bucket in &self.buckets {
            let guard = bucket
                .read()
                .map_err(|_| anyhow::anyhow!("Catapult bucket lock poisoned"))?;
            flat.extend(guard.iter().copied());
            offsets.push(flat.len() as u32);
        }

        let mut raw = Vec::<u8>::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&(self.hashes as u32).to_le_bytes());
        raw.extend_from_slice(&(self.capacity as u32).to_le_bytes());
        raw.extend_from_slice(&self.seed.to_le_bytes());
        raw.extend_from_slice(&(bucket_count as u32).to_le_bytes());
        raw.extend_from_slice(&(flat.len() as u32).to_le_bytes());
        for x in offsets {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        for id in flat {
            raw.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(path, raw)?;
        Ok(())
    }

    fn load_snapshot(&self, path: &std::path::Path) -> anyhow::Result<()> {
        const MAGIC: &[u8; 8] = b"GICAT001";
        let raw = std::fs::read(path)?;
        if raw.len() < 32 || &raw[0..8] != MAGIC {
            anyhow::bail!("invalid Catapult snapshot header");
        }
        let hashes = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let capacity = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let seed = u64::from_le_bytes(raw[16..24].try_into()?);
        let bucket_count = u32::from_le_bytes(raw[24..28].try_into()?) as usize;
        let total = u32::from_le_bytes(raw[28..32].try_into()?) as usize;
        if hashes != self.hashes
            || capacity != self.capacity
            || seed != self.seed
            || bucket_count != self.buckets.len()
        {
            anyhow::bail!(
                "Catapult snapshot config mismatch: hashes={}/{}, capacity={}/{}, seed={}/{}, buckets={}/{}",
                hashes, self.hashes,
                capacity, self.capacity,
                seed, self.seed,
                bucket_count, self.buckets.len()
            );
        }
        let offsets_bytes = (bucket_count + 1)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("snapshot size overflow"))?;
        let ids_bytes = total
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("snapshot size overflow"))?;
        let expected = 32usize
            .checked_add(offsets_bytes)
            .and_then(|x| x.checked_add(ids_bytes))
            .ok_or_else(|| anyhow::anyhow!("snapshot size overflow"))?;
        if raw.len() != expected {
            anyhow::bail!(
                "Catapult snapshot byte length mismatch: got {}, expected {}",
                raw.len(), expected
            );
        }

        let mut offsets = Vec::<u32>::with_capacity(bucket_count + 1);
        let mut off = 32usize;
        for _ in 0..=bucket_count {
            offsets.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(total as u32)
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            anyhow::bail!("invalid Catapult snapshot offsets");
        }
        let mut ids = Vec::<u32>::with_capacity(total);
        for _ in 0..total {
            ids.push(u32::from_le_bytes(raw[off..off + 4].try_into()?));
            off += 4;
        }

        for b in 0..bucket_count {
            let lo = offsets[b] as usize;
            let hi = offsets[b + 1] as usize;
            if hi - lo > self.capacity {
                anyhow::bail!("snapshot bucket {} exceeds capacity", b);
            }
            let mut guard = self.buckets[b]
                .write()
                .map_err(|_| anyhow::anyhow!("Catapult bucket lock poisoned"))?;
            guard.clear();
            guard.extend(ids[lo..hi].iter().copied());
        }
        Ok(())
    }

    fn resident_entries(&self) -> anyhow::Result<usize> {
        let mut total = 0usize;
        for bucket in &self.buckets {
            let guard = bucket
                .read()
                .map_err(|_| anyhow::anyhow!("Catapult bucket lock poisoned"))?;
            total += guard.len();
        }
        Ok(total)
    }

'''
    s = once(s, marker, helper + marker, "Catapult snapshot methods")

    s = once(
        s,
        """    let catapult_config = PaperCatapultConfig::from_env()?;

    let portal_router = match std::env::var_os("DISKANN_IP_PORTAL_ROUTER_FILE") {
""",
        """    let catapult_config = PaperCatapultConfig::from_env()?;
    let catapult_snapshot_load = std::env::var_os("DISKANN_CATAPULT_SNAPSHOT_LOAD");
    let catapult_snapshot_dump = std::env::var_os("DISKANN_CATAPULT_SNAPSHOT_DUMP");
    let catapult_freeze = std::env::var("DISKANN_CATAPULT_FREEZE")
        .ok()
        .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "FALSE"))
        .unwrap_or(false);

    let portal_router = match std::env::var_os("DISKANN_IP_PORTAL_ROUTER_FILE") {
""",
        "Catapult snapshot env",
    )

    s = once(
        s,
        """        let catapult = catapult_config.map(|cfg| {
            PaperCatapult::new(queries.ncols(), searcher.medoid(), cfg)
        });

        let start = Instant::now();
""",
        """        let catapult = catapult_config.map(|cfg| {
            PaperCatapult::new(queries.ncols(), searcher.medoid(), cfg)
        });
        if let (Some(c), Some(path)) = (catapult.as_ref(), catapult_snapshot_load.as_ref()) {
            c.load_snapshot(std::path::Path::new(path))?;
            eprintln!(
                "Loaded frozen Catapult snapshot with {} resident entries",
                c.resident_entries()?
            );
        }

        let start = Instant::now();
""",
        "load Catapult snapshot",
    )

    s = once(
        s,
        """                        if let (Some(c), Some((Some(bucket), _))) = (catapult.as_ref(), routed.as_ref()) {
                            if base_count > 0 {
                                if let Err(e) = c.insert(*bucket, id_chunk[0]) {
""",
        """                        if !catapult_freeze {
                            if let (Some(c), Some((Some(bucket), _))) = (catapult.as_ref(), routed.as_ref()) {
                                if base_count > 0 {
                                    if let Err(e) = c.insert(*bucket, id_chunk[0]) {
""",
        "freeze Catapult updates start",
    )

    s = once(
        s,
        """                                    has_any_search_failed.store(true, Ordering::Release);
                                }
                            }
                        }
""",
        """                                        has_any_search_failed.store(true, Ordering::Release);
                                    }
                                }
                            }
                        }
""",
        "freeze Catapult updates close",
    )

    s = once(
        s,
        """        if let Some(c) = catapult.as_ref() {
            let (usage, starts) = c.metrics();
            search_result.catapult_usage_percentage = usage;
            search_result.mean_catapult_starts = starts;
        }

        l_span.end();
""",
        """        if let Some(c) = catapult.as_ref() {
            let (usage, starts) = c.metrics();
            search_result.catapult_usage_percentage = usage;
            search_result.mean_catapult_starts = starts;
            if let Some(path) = catapult_snapshot_dump.as_ref() {
                c.save_snapshot(std::path::Path::new(path))?;
                eprintln!(
                    "Dumped Catapult snapshot with {} resident entries",
                    c.resident_entries()?
                );
            }
        }

        l_span.end();
""",
        "dump Catapult snapshot",
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
    patch_provider_waypoint(provider)
    patch_catapult_benchmark(benchmark)
    patch_waypoint_benchmark(benchmark)
    patch_snapshot(benchmark)
    print(f"patched DiskANN {PINNED} with waypoint cache and Catapult snapshots")


if __name__ == "__main__":
    main()
