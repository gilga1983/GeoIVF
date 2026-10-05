#!/usr/bin/env python3
"""Patch pinned DiskANN to support a workload-selected static full-node cache.

Set DISKANN_STATIC_CACHE_IDS_FILE to a GIDST001 uint32-ID file. The benchmark
then preloads exactly those graph nodes into DiskANN's existing Cache<Data>
(vector + adjacency + associated data) instead of choosing BFS nodes.

Search logic and cache-hit behavior are otherwise upstream DiskANN.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def once(s: str, old: str, new: str, label: str) -> str:
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    return s.replace(old, new)


def patch_cache_enum(path: Path) -> None:
    s = path.read_text()
    s = once(
        s,
        """#[derive(PartialEq)]
pub enum CachingStrategy {
    None,
    StaticCacheWithBfsNodes(usize),
}
""",
        """#[derive(PartialEq)]
pub enum CachingStrategy {
    None,
    StaticCacheWithBfsNodes(usize),
    /// Preload exactly the database IDs stored in a GIDST001 file.
    StaticCacheWithIdFile(String),
}
""",
        "cache strategy enum",
    )
    path.write_text(s)


def patch_factory(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(_) => match self.cache {
                Some(ref cache) => CachedDiskVertexProvider::new(
                    header,
                    max_batch_size,
                    sector_reader,
                    cache.clone(),
                ),
                None => Err(diskann_error!(
                    ErrorKind::IndexError,
                    "Cache must be initialised for StaticCacheWithBfsNodes caching strategy",
                )),
            },
            CachingStrategy::None => CachedDiskVertexProvider::new(
""",
        """        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(_)
            | CachingStrategy::StaticCacheWithIdFile(_) => match self.cache {
                Some(ref cache) => CachedDiskVertexProvider::new(
                    header,
                    max_batch_size,
                    sector_reader,
                    cache.clone(),
                ),
                None => Err(diskann_error!(
                    ErrorKind::IndexError,
                    "Cache must be initialised for static caching strategy",
                )),
            },
            CachingStrategy::None => CachedDiskVertexProvider::new(
""",
        "cached provider match",
    )

    old = """            CachingStrategy::None => {}
        }

        info!("Cache setup took: {} ms", timer.elapsed().as_millis());
"""
    new = """            CachingStrategy::StaticCacheWithIdFile(ref path) => {
                let ids = Self::load_static_cache_ids(std::path::Path::new(path))?;
                if ids.is_empty() {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "static cache ID file is empty",
                    ));
                }
                let graph_metadata = self.get_header()?;
                let graph_metadata = graph_metadata.metadata();
                if ids.iter().any(|id| (*id as u64) >= graph_metadata.num_pts) {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "static cache ID outside graph range",
                    ));
                }
                self.cache = Some(Arc::new(self.build_cache_from_ids(
                    &ids,
                    graph_metadata.dims,
                )?));
            }
            CachingStrategy::None => {}
        }

        info!("Cache setup took: {} ms", timer.elapsed().as_millis());
"""
    s = once(s, old, new, "file cache setup arm")

    marker = """    fn build_cache_via_bfs(
"""
    helper = r'''    fn load_static_cache_ids(path: &std::path::Path) -> ANNResult<Vec<u32>> {
        let raw = std::fs::read(path).map_err(|e| {
            diskann_error!(
                ErrorKind::IndexError,
                "failed reading static cache ID file {}: {}",
                path.display(),
                e,
            )
        })?;
        const MAGIC: &[u8; 8] = b"GIDST001";
        if raw.len() < 16 || &raw[0..8] != MAGIC {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid static cache ID header",
            ));
        }
        let count = u32::from_le_bytes(raw[8..12].try_into().map_err(|_| {
            diskann_error!(ErrorKind::IndexError, "invalid static cache count")
        })?) as usize;
        let reserved = u32::from_le_bytes(raw[12..16].try_into().map_err(|_| {
            diskann_error!(ErrorKind::IndexError, "invalid static cache reserved field")
        })?);
        let expected = 16usize
            .checked_add(count.checked_mul(4).ok_or_else(|| {
                diskann_error!(ErrorKind::IndexError, "static cache size overflow")
            })?)
            .ok_or_else(|| diskann_error!(ErrorKind::IndexError, "static cache size overflow"))?;
        if reserved != 0 || raw.len() != expected {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid static cache ID file size",
            ));
        }

        let mut ids = Vec::with_capacity(count);
        let mut seen = HashSet::with_capacity(count);
        let mut off = 16usize;
        for _ in 0..count {
            let id = u32::from_le_bytes(raw[off..off + 4].try_into().map_err(|_| {
                diskann_error!(ErrorKind::IndexError, "invalid static cache ID")
            })?);
            off += 4;
            if !seen.insert(id) {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "duplicate ID in static cache file",
                ));
            }
            ids.push(id);
        }
        Ok(ids)
    }

    fn build_cache_from_ids(
        &self,
        ids: &[u32],
        dimension: usize,
    ) -> ANNResult<Cache<Data>> {
        info!("Building workload-selected cache with {} nodes.", ids.len());
        let mut cache = Cache::new(dimension, ids.len())?;
        let mut vertex_provider =
            self.create_disk_vertex_provider(BEAM_WIDTH_FOR_BFS, &self.get_header()?)?;

        for chunk in ids.chunks(BEAM_WIDTH_FOR_BFS) {
            vertex_provider.load_vertices(chunk)?;
            for (idx, node) in chunk.iter().enumerate() {
                Self::insert_in_cache(node, idx, &mut vertex_provider, &mut cache)?;
            }
            vertex_provider.clear();
        }
        Ok(cache)
    }

'''
    s = once(s, marker, helper + marker, "file cache builder")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s = path.read_text()
    old = """    let caching_strategy = if let Some(num_nodes) = search_params.num_nodes_to_cache {
        CachingStrategy::StaticCacheWithBfsNodes(num_nodes)
    } else {
        CachingStrategy::None
    };
"""
    new = """    let caching_strategy = if let Some(path) =
        std::env::var_os("DISKANN_STATIC_CACHE_IDS_FILE")
    {
        CachingStrategy::StaticCacheWithIdFile(path.to_string_lossy().into_owned())
    } else if let Some(num_nodes) = search_params.num_nodes_to_cache {
        CachingStrategy::StaticCacheWithBfsNodes(num_nodes)
    } else {
        CachingStrategy::None
    };
"""
    s = once(s, old, new, "benchmark cache strategy")
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    cache = root / "diskann-disk/src/data_model/cache.rs"
    factory = root / "diskann-disk/src/search/provider/disk_vertex_provider_factory.rs"
    benchmark = root / "diskann-benchmark/src/disk_index/search.rs"
    for path in (cache, factory, benchmark):
        if not path.is_file():
            raise SystemExit(f"unexpected DiskANN checkout: missing {path}")

    patch_cache_enum(cache)
    patch_factory(factory)
    patch_benchmark(benchmark)
    print("patched DiskANN with workload-selected full-node static cache")


if __name__ == "__main__":
    main()
