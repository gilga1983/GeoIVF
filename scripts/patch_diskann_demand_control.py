#!/usr/bin/env python3
"""Add experiment-only recent-set controls after the consolidated NavHints patch.

Modes preserve the same 512-ID PQ scan and single best extra start:
  fifo      - deployed Recent512
  history   - uniform reservoir over unique prior successful destinations
  randomdb  - fixed deterministic random corpus IDs
"""
from __future__ import annotations
import argparse
from pathlib import Path


def once(s: str, old: str, new: str, label: str) -> str:
    n=s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {n}")
    return s.replace(old,new,1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("diskann",type=Path)
    args=ap.parse_args()
    p=args.diskann/"diskann-benchmark/src/disk_index/search.rs"
    s=p.read_text()

    old=r'''    let experience_cache_capacity = std::env::var("DISKANN_EXPERIENCE_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let experience_hub_capacity = std::env::var("DISKANN_EXPERIENCE_HUB_CAPACITY")
'''
    new=r'''    let experience_cache_capacity = std::env::var("DISKANN_EXPERIENCE_CACHE_CAPACITY")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(512);
    let experience_recent_mode =
        std::env::var("DISKANN_EXPERIENCE_RECENT_MODE").unwrap_or_else(|_| "fifo".to_string());
    if !matches!(experience_recent_mode.as_str(), "fifo" | "history" | "randomdb") {
        anyhow::bail!("DISKANN_EXPERIENCE_RECENT_MODE must be fifo, history, or randomdb");
    }
    let experience_random_db_size = std::env::var("DISKANN_EXPERIENCE_RANDOM_DB_SIZE")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let experience_hub_capacity = std::env::var("DISKANN_EXPERIENCE_HUB_CAPACITY")
'''
    s=once(s,old,new,"recent-mode config")

    old=r'''            let mut cache_inserts = 0usize;
            let mut cache_skips = 0usize;
            let mut cache_evictions = 0usize;

            // The persisted page order is the freshness policy: newest direct
'''
    new=r'''            let mut cache_inserts = 0usize;
            let mut cache_skips = 0usize;
            let mut cache_evictions = 0usize;
            let mut history_seen = HashSet::<u32>::new();
            let mut history_unique = 0usize;

            if experience_recent_mode == "randomdb" && experience_cache_capacity > 0 {
                if experience_random_db_size < experience_cache_capacity {
                    anyhow::bail!("random-db universe smaller than recent-set capacity");
                }
                // 104729 is coprime to the PubMed1M universe, so this deterministic
                // arithmetic walk produces unique corpus IDs without RNG state.
                let n = experience_random_db_size as u64;
                for i in 0..experience_cache_capacity {
                    let id = ((314159u64 + (i as u64) * 104729u64) % n) as u32;
                    if !cache_members.insert(id) {
                        anyhow::bail!("random-db control generated duplicate ID");
                    }
                    cache_ids.push(id);
                }
            }

            // The persisted page order is the freshness policy: newest direct
'''
    s=once(s,old,new,"recent-mode state")

    old=r'''                // Update the volatile cache after the completed query, so this
                // request can influence only future requests.
                if experience_cache_capacity > 0 {
                    if cache_members.contains(&winner) {
                        cache_skips += 1;
                    } else {
                        if cache_ids.len() < experience_cache_capacity {
                            cache_ids.push(winner);
                            cache_members.insert(winner);
                        } else {
                            let victim = cache_ids[cache_next];
                            cache_members.remove(&victim);
                            cache_ids[cache_next] = winner;
                            cache_members.insert(winner);
                            cache_next = (cache_next + 1) % experience_cache_capacity;
                            cache_evictions += 1;
                        }
                        cache_inserts += 1;
                    }
                }
'''
    new=r'''                // Update the experiment's 512-ID control after completion, so
                // the current request can influence only future requests.
                if experience_cache_capacity > 0 {
                    match experience_recent_mode.as_str() {
                        "fifo" => {
                            if cache_members.contains(&winner) {
                                cache_skips += 1;
                            } else {
                                if cache_ids.len() < experience_cache_capacity {
                                    cache_ids.push(winner);
                                    cache_members.insert(winner);
                                } else {
                                    let victim = cache_ids[cache_next];
                                    cache_members.remove(&victim);
                                    cache_ids[cache_next] = winner;
                                    cache_members.insert(winner);
                                    cache_next = (cache_next + 1) % experience_cache_capacity;
                                    cache_evictions += 1;
                                }
                                cache_inserts += 1;
                            }
                        }
                        "history" => {
                            if history_seen.insert(winner) {
                                history_unique += 1;
                                if cache_ids.len() < experience_cache_capacity {
                                    cache_ids.push(winner);
                                    cache_members.insert(winner);
                                    cache_inserts += 1;
                                } else {
                                    // Deterministic reservoir sampling over unique historical
                                    // destinations. The hash uses only information available
                                    // after this completed query.
                                    let mut x = (winner as u64)
                                        ^ (qi as u64).wrapping_mul(0x9E3779B97F4A7C15);
                                    x ^= x >> 30;
                                    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                                    x ^= x >> 27;
                                    x = x.wrapping_mul(0x94D049BB133111EB);
                                    x ^= x >> 31;
                                    let slot = (x % history_unique as u64) as usize;
                                    if slot < experience_cache_capacity {
                                        let victim = cache_ids[slot];
                                        cache_members.remove(&victim);
                                        cache_ids[slot] = winner;
                                        cache_members.insert(winner);
                                        cache_evictions += 1;
                                        cache_inserts += 1;
                                    }
                                }
                            } else {
                                cache_skips += 1;
                            }
                        }
                        "randomdb" => {}
                        _ => unreachable!(),
                    }
                }
'''
    s=once(s,old,new,"recent-mode update")
    p.write_text(s)


if __name__=="__main__":
    main()
