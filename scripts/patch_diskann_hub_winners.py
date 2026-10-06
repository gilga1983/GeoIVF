#!/usr/bin/env python3
"""Add 16K-hub aggregated winner hints to vertex-NavHints DiskANN.

Apply after patch_diskann_vertex_navhints.py --variants 8.

Variants 5,6,7 form a 12-ID tail on every node; only the first 10 IDs are
meaningful. Enable with DISKANN_HUB_WINNER_COUNT in {1,2,4,8,10}.

Whenever the currently expanded closest page carries hub winners, score the
selected prefix with the existing PQ LUT and offer all queue-eligible unseen
winners through DiskANN's ordinary fixed-L frontier.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once

BASE_VARIANTS = 5
SLOTS = 4
MAX_WINNERS = 10


def patch_glue(root: Path):
    p = root / "diskann/src/graph/glue.rs"
    s = p.read_text()
    marker = """    /// Score the selected continuation overlay carried by the closest expanded node.
"""
    hook = """    /// Score independent successful winners aggregated at a 16K routing hub.
    fn hub_winner_distances<F>(
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
    s = once(s, marker, hook + marker, "hub winner SearchAccessor hook")
    p.write_text(s)

    p = root / "diskann/src/graph/index.rs"
    s = p.read_text()
    anchor = """                let mut vertex_hint_best: Option<(A::Id, f32)> = None;
"""
    insert = r'''                let mut hub_candidates: Vec<(A::Id, f32)> = Vec::new();
                accessor
                    .hub_winner_distances(&scratch.beam_nodes, |id, distance| {
                        if !scratch.visited.contains(&id) {
                            hub_candidates.push((id, distance));
                        }
                    })
                    .await?;
                hub_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (id, distance) in hub_candidates {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }

'''
    s = once(s, anchor, insert + anchor, "hub winner frontier insertion")
    p.write_text(s)


def patch_provider(path: Path):
    s = path.read_text()
    s = once(
        s,
        """    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
}
""",
        """    vertex_hint_last: [u32; 4],
    vertex_hint_last_valid: bool,
    hub_winner_count: usize,
    hub_winner_last: [u32; 10],
    hub_winner_last_valid: bool,
}
""",
        "hub winner accessor state",
    )
    s = once(
        s,
        """            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
        })
""",
        """            vertex_hint_last: [u32::MAX; 4],
            vertex_hint_last_valid: false,
            hub_winner_count: std::env::var("DISKANN_HUB_WINNER_COUNT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            hub_winner_last: [u32::MAX; 10],
            hub_winner_last_valid: false,
        })
""",
        "hub winner accessor init",
    )

    marker = """    fn vertex_hint_distances<F>(
"""
    method = r'''    fn hub_winner_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let count = self.hub_winner_count;
            if count == 0 || expanded.is_empty() {
                return Ok(());
            }
            if !matches!(count, 1 | 2 | 4 | 8 | 10) {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "DISKANN_HUB_WINNER_COUNT must be one of 1,2,4,8,10",
                ));
            }

            let associated = self
                .scratch
                .vertex_provider
                .get_associated_data(&expanded[0])?;
            let all = Data::embedded_navigation_hints(associated);
            const BASE: usize = 5 * 4;
            const MAX: usize = 10;
            if all.len() < BASE + 12 {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "hub-winner associated-data shape mismatch",
                ));
            }
            let winners = &all[BASE..BASE + MAX];

            let mut signature = [u32::MAX; MAX];
            signature.copy_from_slice(winners);
            if self.hub_winner_last_valid && signature == self.hub_winner_last {
                return Ok(());
            }
            self.hub_winner_last = signature;
            self.hub_winner_last_valid = true;

            let mut ids = [u32::MAX; MAX];
            let mut n = 0usize;
            for &id in &winners[..count] {
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
    s = once(s, marker, method + marker, "hub winner accessor scoring")
    path.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diskann", type=Path)
    args = ap.parse_args()
    root = args.diskann.resolve()
    patch_glue(root)
    patch_provider(root / "diskann-disk/src/search/provider/disk_provider.rs")
    print("patched DiskANN with 16K-hub aggregated independent winners")


if __name__ == "__main__":
    main()
