#!/usr/bin/env python3
"""Source-level CatapultDB parity audit. Fail closed on unexpected source drift.

Input #1 is the pinned ORIGINAL author Rust checkout.
Input #2 is the original DiskANN port Python script (NOT the aligned variant).
Input #3 is the optionally patched Microsoft DiskANN test checkout.

This is an independent evidence/audit artifact, NOT a proof of end-to-end
equivalence between in-memory Catapult and SSD DiskANN.
"""
import argparse
import json
import subprocess
from pathlib import Path


def require(label,s,fragment):
    if fragment not in s:
        raise AssertionError(f"{label}: expected evidence missing: {fragment[:125]}")

def audit(author,ported,patched=None):
    assert (author/"src/search/adjacency_graph.rs").is_file()
    src=(author/"src/search/adjacency_graph.rs").read_text()
    orig_hasher=(author/"src/search/hash_start/hyperplane_hasher.rs").read_text()
    orig_starter=(author/"src/search/hash_start/engine_starter.rs").read_text()
    orig_lru=(author/"src/sets/catapults/lru_set.rs").read_text()
    orig_bench=(author/"src/bin/run_queries.rs").read_text()
    port=ported.read_text()
    author_sha=subprocess.check_output(["git","-C",str(author),"rev-parse","HEAD"],text=True).strip()
    assert author_sha=="a16eddd34b4339db5ec86e292470ce7929179bc3",author_sha

    requirements={
        "author_has_random_hyperplane_LSH":(orig_hasher,"rng.sample_iter(StandardNormal)"),
        "author_packs_signature_MSB_first":(orig_hasher,"projected = projected << 1"),
        "author_uses_LRU":(orig_lru,"self.queue.remove(pos)"),
        "author_eviction_removes_oldest":(orig_lru,"self.queue.pop_front()"),
        "author_stores_winning_results":(src,".new_catapult(hash_search.signature, best_result)"),
        "author_always_starts_from_medoid":(src,"hash_search.starting_node"),
        "author_uses_all_candidate_starts":(src,"distances.shrink_to(k)"),
        "author_start_frontier_bounded_by_beam":(src,"SmallestKCandidates::new(beam_width)"),
        "author_LSH_table_thread_safe":(orig_starter,"RwLock::new(T::new(params.bucket_capacity))"),
        "author_parallel_original_benchmark":(orig_bench,"thread::spawn"),
        "port_uses_random_hyperplanes":(port,"// Random-hyperplane LSH needs isotropic normal directions."),
        "port_preserves_medoid_entry":(port,"starts.push(base_start)"),
        "port_stores_winner":(port,"c.insert(*bucket, id_chunk[0])"),
        "port_LRU_duplicate_refresh":(port,"guard.remove(pos)"),
        "port_LRU_evicts_oldest":(port,"guard.pop_front()"),
        "port_buckets_have_locks":(port,"RwLock<VecDeque<u32>>"),
        "port_hash_bits_default_8":(port,'unwrap_or_else(|_| "8".to_string())'),
        "port_bucket_capacity_default_40":(port,'unwrap_or_else(|_| "40".to_string())'),
    }
    for label,(source,fragment) in requirements.items(): require(label,source,fragment)
    flags={
       "bit_packing_differs":("code |= 1usize << h" in port) and
           ("projected = projected << 1" in orig_hasher),
       "gaussian_generator_differs":("Box-Muller transform" in port) and
           ("sample_iter(StandardNormal)" in orig_hasher),
       "original_port_skips_medoid_winner":(
           "if destination == self.medoid" in port) and
           ("if new_cata == self.starting_node" not in orig_starter),
       "original_port_times_before_bucket_insert":(
           port.find("stats.total_execution_time_us = query_timer.elapsed().as_micros();")
             < port.find("c.insert(*bucket, id_chunk[0])")),
       "authors_shrink_to_does_not_truncate_starts":(
           "distances.shrink_to(k)" in src),
       "underlying_disk_engine_uses_PQ_instead_of_full_vectors":(
           "fn pq_distances" not in port and
           "search_with_start_points" in port and
           "starting_point.payload.l2_squared(query)" in src),
       "original_port_inherits_parallel_benchmark":(
           "let pool = create_thread_pool(search_params.num_threads)" in port),
    }
    for key,v in flags.items():
        if not v: raise AssertionError(f"expected fidelity issue disappeared: {key}")

    amended={}
    if patched is not None:
        s=(patched/"diskann-benchmark/src/disk_index/search.rs").read_text()
        cargo=(patched/"diskann-benchmark/Cargo.toml").read_text()
        amended={
          "new_source_uses_author_Gaussian":"sample_iter(StandardNormal)" in s,
          "new_source_uses_author_MSB_hash":"code = (code << 1) | usize::from(dot >= 0.0)" in s,
          "new_source_does_not_skip_medoid":"if destination == self.medoid" not in s,
          "new_source_imports_rnd_distr":"rand_distr.workspace = true" in cargo,
          "new_source_times_after_insert":(
              s.find("stats.total_execution_time_us = query_timer.elapsed().as_micros();")
                >s.find("c.insert(*bucket, id_chunk[0])")),
          "new_source_still_uses_native_PQ":"search_with_start_points" in s,
        }
        if not all(amended.values()):
            raise AssertionError(f"candidate author-aligned patch is incomplete: {amended}")

    return {
      "status":"source-parity audit completed; native oracle separately verifies runtime",
      "author_repository":"MRandl/catapult-db",
      "author_git_sha":author_sha,
      "port":"scripts/patch_diskann_paper_catapult.py on pinned Microsoft DiskANN3",
      "verified_source_conditions":list(requirements),
      "discrepancies":flags,
      "candidate_author_aligned_source_checks":amended,
      "invariants_matching":["LSH bucket stores recent query winners",
                            "always includes medoid starting point",
                            "per-bucket duplicate refreshes MRU",
                            "bounded bucket length & oldest-entry eviction",
                            "result winner inserted into query's bucket",
                            "start IDs passed to native search procedure"],
      "qualification":"A comparison that scores SSD I/O is more controlled than comparing in-memory Catapult against disk-based NavHints, but the baseline should either use author-aligned policy semantics or be called a close independent adaptation.",
      "caution":"The original author evaluates exact full vectors on an in-memory graph. The port must use PQ-based DiskANN graph traversal and may not have identical outputs. Query parallelism in the benchmark can also affect online update ordering, even when all calls are thread-safe.",
    }


def markdown(r):
    v=r["discrepancies"]
    s=["# Original CatapultDB versus our DiskANN implementation: source audit","",
       "Pinned original author implementation: "+r["author_git_sha"],"",
       "## What is correctly reproduced",
       "We have the same paper mechanism: 8-bit random-hyperplane LSH,"
       " 40 IDs per bucket, deduplicating LRU with per-bucket locks, dynamic"
       " feedback of the returned winner, and query-time starting points"
       " containing the native graph medoid plus cached IDs.",
       "",
       "## Differences that require disclosure or an author-aligned ablation",
       "| Issue | Current source result | Significance |",
       "|---|---|---|",
       f"| Random Gaussian source | {'Different Box-Muller vs StandardNormal' if v['gaussian_generator_differs'] else 'Same'} | Same LSH family, not seed-identical |",
       f"| Signature bit order | {'Reversed' if v['bit_packing_differs'] else 'Same'} | Mere permutation of bucket labels for fixed planes |",
       f"| Cache successful medoid | {'Skipped by our port' if v['original_port_skips_medoid_winner'] else 'Admitted'} | Real bucket state and LRU occupancy difference |",
       f"| Count update overhead in request time | {'No' if v['original_port_times_before_bucket_insert'] else 'Yes'} | Catapult timing was favorably biased |",
       f"| Initial candidate count | {'Original shrink_to DOES NOT truncate' if v['authors_shrink_to_does_not_truncate_starts'] else 'Different'} | Need preserve effective beam budget, not misunderstand API |",
       f"| Original graph vs SSD port | {'Exact-vector RAM vs PQ SSD' if v['underlying_disk_engine_uses_PQ_instead_of_full_vectors'] else 'Same'} | Necessary systems adaptation, not parity |",
       f"| Online execution | {'Parallel benchmark calls' if v['original_port_inherits_parallel_benchmark'] else 'Serial'} | Thread safety does not imply chronological replay |",
       "",
       "## Recommended baseline naming",
       "Use **CatapultDB (adapted to DiskANN)** after author-alignment corrections;"
       " label the current published source as an **independent CatapultDB-style adaptation** until verified.",
       "Do not claim original-author in-memory latency is directly comparable"
       " to disk-resident NavHints SSD latency.",
       "",
       "The audit distinguishes source conformance from end-to-end algorithm"
       " equivalence. Additional differential tests are required before"
       " accepting a corrected benchmark.",
       ""]
    return "\n".join(s)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author",required=True,type=Path)
    p.add_argument("--port",required=True,type=Path)
    p.add_argument("--patched",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=audit(args.author,args.port,args.patched)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"source-audit.json").write_text(json.dumps(r,indent=2)+"\n")
    report=markdown(r)
    (args.out/"source-audit.md").write_text(report)
    print("CATAPULT_SOURCE_PARITY_AUDIT_COMPLETED",flush=True)
    print(report,flush=True)

if __name__=="__main__":main()
