#!/usr/bin/env python3
"""Exploratory source-level selector-width ablation on a pinned DiskANN checkout.

Apply after:
* NavHints: patch_diskann_hint_ivf_packed_direct.py, patch_diskann_experience_core.py
* Catapult: patch_diskann_catapult_snapshot.py,
  align_catapult_author_semantics.py, patch_catapult_author_causal_eval.py

No policy state size, graph, search L, beam, or recall calculation changes.
NavHints already scores ten best IDs from 512 recent results. Emit the best
1/2/4/8/10 instead of only one, preserving old behavior exactly at 1.
Catapult can offer medoid plus the most recently used 1/4/16 buckets IDs, or
all bucket IDs (original). This latter mode is diagnostic, not the released
CatapultDB algorithm.
All caps are read once per process. Intended for I/O/recall mechanisms only:
do not treat this run as an end-to-end latency optimization comparison.
"""
from pathlib import Path
import argparse


def once(s, before, after, tag):
    count = s.count(before)
    if count != 1:
        raise RuntimeError(f"{tag}: expected unique exact source anchor, found {count}")
    return s.replace(before, after, 1)


def navhints(root):
    f = root / "diskann-disk/src/search/provider/disk_provider.rs"
    s = f.read_text()
    before = '''                    if let Some(&(distance, id)) = top_cache.first() {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
'''
    after = '''                    // Diagnostic selector width: top-K from already-scored
                    // Recent512 IDs, no new PQ scans or state. The default K=1
                    // preserves the prior one-destination injection exactly.
                    static RECENT_START_CAP: std::sync::OnceLock<usize> =
                        std::sync::OnceLock::new();
                    let recent_cap = *RECENT_START_CAP.get_or_init(|| {
                        std::env::var("DISKANN_DIAG_NAV_RECENT_STARTS")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(1)
                            .min(10)
                    });
                    for &(distance, id) in top_cache.iter().take(recent_cap) {
                        if id != winner.1 {
                            emitted += 1;
                            f(id, distance);
                        }
                    }
'''
    s = once(s, before, after, "NavHints recent-result seed count")
    f.write_text(s)
    print("DIAGNOSTIC_NAV_TOPK_PATCH_APPLIED", flush=True)


def catapult(root):
    f = root / "diskann-benchmark/src/disk_index/search.rs"
    s = f.read_text()
    before = '''        for &id in guard.iter() {
            if id != base_start && !starts.contains(&id) {
                starts.push(id);
            }
        }
'''
    after = '''        // Diagnostic candidate cap, NOT a change to the released author's
        // default. Limited modes keep newest per-bucket IDs (LRU order).
        // Unlimited mode preserves old iteration order and all candidates.
        static CATAPULT_START_CAP: std::sync::OnceLock<usize> =
            std::sync::OnceLock::new();
        let cap = *CATAPULT_START_CAP.get_or_init(|| {
            std::env::var("DISKANN_DIAG_CATAPULT_BUCKET_STARTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
        if cap == usize::MAX {
            for &id in guard.iter() {
                if id != base_start && !starts.contains(&id) {
                    starts.push(id);
                }
            }
        } else {
            for &id in guard.iter().rev().take(cap) {
                if id != base_start && !starts.contains(&id) {
                    starts.push(id);
                }
            }
        }
'''
    s = once(s, before, after, "Catapult offered bucket seeds")
    f.write_text(s)
    print("DIAGNOSTIC_CATAPULT_BUCKET_CAP_PATCH_APPLIED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["nav", "cat"])
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.backend == "nav":
        navhints(root)
    else:
        catapult(root)


if __name__ == "__main__":
    main()
