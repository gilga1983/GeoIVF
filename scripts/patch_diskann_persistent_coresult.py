#!/usr/bin/env python3
"""Add an optional persistent co-result overlay to vertex-NavHints DiskANN.

Apply after:
  patch_diskann_vertex_navhints.py --variants 6
  patch_diskann_semantic_cache_page.py

The persistent overlay is another 4-ID slice in associated data. Unlike the
query-specific semantic-cache siblings, it is available whenever its vertex
page is reached, even if that vertex is no longer in the RAM value cache.

Enable with:
  DISKANN_PERSISTENT_HINT_VARIANT=5

At each natural beam, score the four persistent IDs attached to the closest
expanded vertex using the existing PQ LUT and admit at most the best candidate
through DiskANN's ordinary fixed-L gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from patch_diskann_start_points import once


def patch_glue(root: Path) -> None:
    p = root / "diskann/src/graph/glue.rs"
    s = p.read_text()
    marker = """    /// Score semantic-cache siblings once their cached anchor has actually been expanded.
"""
    hook = """    /// Score persistent page-resident co-result hints.
    fn persistent_hint_distances<F>(
        &mut self,
        _expanded: &[Self::Id],
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    s = once(s, marker, hook + marker, "persistent SearchAccessor hook")
    p.write_text(s)

    p = root / "diskann/src/graph/index.rs"
    s = p.read_text()
    anchor = """                let mut semantic_candidates: Vec<(A::Id, f32)> = Vec::new();
"""
    insert = r'''                let mut persistent_best: Option<(A::Id, f32)> = None;
                accessor
                    .persistent_hint_distances(&scratch.beam_nodes, |id, distance| {
                        if scratch.visited.contains(&id) {
                            return;
                        }
                        let better = persistent_best
                            .is_none_or(|current| distance.total_cmp(&current.1).is_lt());
                        if better {
                            persistent_best = Some((id, distance));
                        }
                    })
                    .await?;
                if let Some((id, distance)) = persistent_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }

'''
    s = once(s, anchor, insert + anchor, "persistent candidate insertion")
    p.write_text(s)


def patch_provider(path: Path) -> None:
    s = path.read_text()

    s = once(
        s,
        """    semantic_cache_entry: Option<&'a [u32]>,
    semantic_cache_siblings_emitted: bool,
}
""",
        """    semantic_cache_entry: Option<&'a [u32]>,
    semantic_cache_siblings_emitted: bool,
    persistent_hint_variant: Option<usize>,
    persistent_hint_last: [u32; 4],
    persistent_hint_last_valid: bool,
}
""",
        "persistent accessor state",
    )

    s = once(
        s,
        """            semantic_cache_entry: strategy.semantic_cache_entry,
            semantic_cache_siblings_emitted: false,
        })
""",
        """            semantic_cache_entry: strategy.semantic_cache_entry,
            semantic_cache_siblings_emitted: false,
            persistent_hint_variant: std::env::var("DISKANN_PERSISTENT_HINT_VARIANT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            persistent_hint_last: [u32::MAX; 4],
            persistent_hint_last_valid: false,
        })
""",
        "persistent accessor init",
    )

    marker = """    fn semantic_cache_distances<F>(
"""
    method = r'''    fn persistent_hint_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let Some(variant) = self.persistent_hint_variant else {
                return Ok(());
            };
            if expanded.is_empty() {
                return Ok(());
            }
            const SLOTS: usize = 4;

            let associated = self
                .scratch
                .vertex_provider
                .get_associated_data(&expanded[0])?;
            let all = Data::embedded_navigation_hints(associated);
            let lo = variant
                .checked_mul(SLOTS)
                .ok_or_else(|| diskann_error!(
                    ErrorKind::IndexError,
                    "persistent variant offset overflow",
                ))?;
            if lo + SLOTS > all.len() {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "persistent variant outside associated data",
                ));
            }
            let map = &all[lo..lo + SLOTS];

            let mut signature = [u32::MAX; SLOTS];
            signature.copy_from_slice(map);
            if self.persistent_hint_last_valid && signature == self.persistent_hint_last {
                return Ok(());
            }
            self.persistent_hint_last = signature;
            self.persistent_hint_last_valid = true;

            let mut ids = [u32::MAX; SLOTS];
            let mut n = 0usize;
            for &id in map {
                if id != u32::MAX {
                    ids[n] = id;
                    n += 1;
                }
            }
            if n == 0 {
                return Ok(());
            }

            self.io_tracker.routing_comparisons.fetch_add(
                n,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(&ids[..n], |distance, id| f(id, distance))
        })();
        std::future::ready(result)
    }

'''
    s = once(s, marker, method + marker, "persistent accessor scorer")
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    patch_glue(root)
    patch_provider(root / "diskann-disk/src/search/provider/disk_provider.rs")
    print("patched DiskANN with persistent page-resident co-result hints")


if __name__ == "__main__":
    main()
