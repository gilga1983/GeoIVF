#!/usr/bin/env python3
"""Patch the official CatapultDB artifact with an optional GeoIVF portal starter.

The patch changes only EngineStarter's source of initial graph vertices. With no
CATAPULT_PORTAL_ROUTER_FILE environment variable, behavior is byte-for-byte the
same logic as upstream HEAD. Catapult insertion, eviction, graph expansion, and
beam search remain in the authors' code unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PINNED = "7d473050e0a69079d8fc17158f2967486e54380b"


def once(s: str, old: str, new: str, label: str) -> str:
    if s.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {s.count(old)}")
    return s.replace(old, new, 1)


def patch_engine(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """use std::collections::HashMap;
use std::sync::RwLock;

use crate::{numerics::AlignedBlock, search::hash_start::hasher::SimilarityHasher};
""",
        """use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::{
    numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike},
    search::hash_start::hasher::SimilarityHasher,
};
""",
        "imports",
    )

    s = once(
        s,
        """pub struct EngineStarter {
    hasher: SimilarityHasher,
    cached_starters: RwLock<HashMap<u64, Vec<usize>>>,
    max_len: usize,
}
""",
        r'''#[derive(Debug)]
struct PortalRouter {
    nlist: usize,
    blocks_per_vector: usize,
    centers: Vec<AlignedBlock>,
    portal_ids: Vec<usize>,
    portal_vectors: Vec<AlignedBlock>,
    nprobe: usize,
}

impl PortalRouter {
    fn load(path: &Path, expected_dim: usize, graph_size: usize, nprobe: usize) -> Self {
        assert!(expected_dim > 0 && expected_dim.is_multiple_of(SIMD_LANECOUNT));
        assert!(nprobe > 0 && nprobe <= 32);

        let raw = std::fs::read(path).expect("failed to read GeoIVF portal router");
        assert!(raw.len() >= 16, "portal router is truncated");
        assert_eq!(&raw[0..8], b"GIPORT01", "bad portal router magic");
        let nlist = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        assert_eq!(dim, expected_dim, "portal router dimension mismatch");
        assert!(nlist > 0, "portal router must contain cells");

        let floats_per_matrix = nlist.checked_mul(dim).expect("router size overflow");
        let expected = 16usize
            .checked_add(floats_per_matrix.checked_mul(4).unwrap())
            .and_then(|x| x.checked_add(nlist.checked_mul(4).unwrap()))
            .and_then(|x| x.checked_add(floats_per_matrix.checked_mul(4).unwrap()))
            .expect("router byte-size overflow");
        assert_eq!(raw.len(), expected, "portal router byte-size mismatch");

        let blocks_per_vector = dim / SIMD_LANECOUNT;
        let mut offset = 16usize;

        fn read_blocks(raw: &[u8], offset: &mut usize, vectors: usize, blocks_per_vector: usize) -> Vec<AlignedBlock> {
            let mut out = Vec::with_capacity(vectors * blocks_per_vector);
            for _ in 0..vectors * blocks_per_vector {
                let mut lane = [0.0f32; SIMD_LANECOUNT];
                for x in lane.iter_mut() {
                    *x = f32::from_le_bytes(raw[*offset..*offset + 4].try_into().unwrap());
                    assert!(x.is_finite(), "nonfinite portal-router coordinate");
                    *offset += 4;
                }
                out.push(AlignedBlock::new(lane));
            }
            out
        }

        let centers = read_blocks(&raw, &mut offset, nlist, blocks_per_vector);
        let mut portal_ids = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            let id = u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap()) as usize;
            assert!(id < graph_size, "portal id outside graph");
            portal_ids.push(id);
            offset += 4;
        }
        let portal_vectors = read_blocks(&raw, &mut offset, nlist, blocks_per_vector);
        assert_eq!(offset, raw.len());

        Self {
            nlist,
            blocks_per_vector,
            centers,
            portal_ids,
            portal_vectors,
            nprobe: nprobe.min(nlist),
        }
    }

    #[inline]
    fn vector<'a>(&self, matrix: &'a [AlignedBlock], row: usize) -> &'a [AlignedBlock] {
        let begin = row * self.blocks_per_vector;
        &matrix[begin..begin + self.blocks_per_vector]
    }

    fn route(&self, query: &[AlignedBlock]) -> usize {
        assert_eq!(query.len(), self.blocks_per_vector);
        let k = self.nprobe;
        let mut best = [(f32::INFINITY, usize::MAX); 32];

        for cell in 0..self.nlist {
            let dist = query.l2_squared(self.vector(&self.centers, cell));
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

        let mut winner = best[0].1;
        let mut winner_dist = f32::INFINITY;
        for &(_, cell) in &best[..k] {
            let dist = query.l2_squared(self.vector(&self.portal_vectors, cell));
            if dist < winner_dist {
                winner_dist = dist;
                winner = cell;
            }
        }
        self.portal_ids[winner]
    }
}

pub struct EngineStarter {
    hasher: SimilarityHasher,
    cached_starters: RwLock<HashMap<u64, Vec<usize>>>,
    max_len: usize,
    portal: Option<PortalRouter>,
}
''',
        "portal router and starter field",
    )

    s = once(
        s,
        """        Self {
            hasher,
            cached_starters: RwLock::new(HashMap::new()),
            max_len: graph_size,
        }
""",
        r'''        let portal = std::env::var_os("CATAPULT_PORTAL_ROUTER_FILE").map(|path| {
            let raw_nprobe = std::env::var("CATAPULT_PORTAL_NPROBE")
                .expect("CATAPULT_PORTAL_NPROBE is required with CATAPULT_PORTAL_ROUTER_FILE");
            let nprobe: usize = raw_nprobe
                .parse()
                .expect("CATAPULT_PORTAL_NPROBE must be an integer");
            PortalRouter::load(Path::new(&path), stored_vectors_dim, graph_size, nprobe)
        });

        Self {
            hasher,
            cached_starters: RwLock::new(HashMap::new()),
            max_len: graph_size,
            portal,
        }
''',
        "constructor portal load",
    )

    s = once(
        s,
        """    pub fn select_starting_points(&self, query: &[AlignedBlock], k: usize) -> Vec<usize> {
        let signature = self.hasher.hash_int(query);
""",
        """    pub fn select_starting_points(&self, query: &[AlignedBlock], k: usize) -> Vec<usize> {
        if let Some(portal) = &self.portal {
            return vec![portal.route(query)];
        }

        let signature = self.hasher.hash_int(query);
""",
        "portal route branch",
    )

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("catapult", type=Path)
    args = ap.parse_args()
    root = args.catapult.resolve()
    engine = root / "src/search/hash_start/engine_starter.rs"
    if not engine.is_file():
        raise SystemExit("unexpected CatapultDB checkout layout")
    patch_engine(engine)
    print(f"patched CatapultDB {PINNED} with optional GeoIVF portal starter")


if __name__ == "__main__":
    main()
