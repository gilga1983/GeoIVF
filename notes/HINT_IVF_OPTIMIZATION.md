# Hint-IVF performance optimization ledger

Purpose: performance engineering is part of the experimental method. Do not
freeze scientific knees (vocabulary, nlist, nprobe, memory budget) from an
implementation with known avoidable routing overhead.

## Fixed science configuration for optimization A/Bs

- PubMed1M / MedCPT, heldout 5000 queries
- training teacher: ordinary DiskANN L=4, first 5000 queries
- 16,000 learned hints
- spherical Hint-IVF, nlist=512, nprobe=8
- online K=L=1, beam=8, 4 pinned query threads
- exact Recall@1
- same SSD device lock
- adjacent variants alternate run order
- semantic gate: recall, I/O, hops must be identical; when adjacent
  integrated variants use the same accounting, comparisons should match too

## Rung 0: reference selector

Reference implementation:
- prepares/scans Hint-IVF outside the graph accessor;
- full coarse sort;
- concatenated fine candidates;
- full fine sort;
- graph traversal prepares/owns its normal search state.

Clean five-run A/B against integrated selector:
- reference median QPS: 12,278.26
- reference median latency: 322.829 us
- reference median CPU: 35.113 us
- reference median PQ preprocess: 14.149 us
- recall: 14.18%
- I/O/query: 2.4194

## Rung 1: integrated sort-based selector

Changes:
- selection runs inside the same DiskAccessor as graph traversal;
- query PQ lookup table is prepared once;
- expensive ID validation moves out of the per-query path;
- selector otherwise preserves full coarse/fine sort semantics.

Same five-run A/B:
- QPS: 12,834.50 (+4.53%)
- latency: 308.666 us (-4.39%)
- CPU: 30.998 us
- PQ preprocess: 7.181 us
- recall: exactly 14.18%
- I/O/query: exactly 2.4194
- hops: exactly 2.4194

## Rung 2: no-sort selector

Changes relative to integrated sort:
- fixed-size top-nprobe insertion for coarse cells;
- no fine candidate vector;
- canonical minimum tracked online;
- selected buckets scored separately;
- canonical ordering is (distance, cell) coarse and (distance, vertex ID) fine.

Clean five-run adjacent A/B against integrated sort:
- integrated-sort median QPS: 12,863.59
- no-sort median QPS: 13,421.09 (+4.33%)
- integrated-sort latency: 308.441 us
- no-sort latency: 295.469 us (-4.21%)
- integrated-sort CPU: 37.753 us
- no-sort CPU: 23.589 us (-37.52%)
- PQ preprocess essentially unchanged: 8.118 -> 8.049 us
- recall: exactly equal at 14.18%
- I/O/query: exactly equal at 2.4194
- hops: exactly equal at 2.4194
- comparisons: exactly equal at 963.1766

The selector knee must therefore be rediscovered after optimization.

## Rung 3: pooled-batch selector (running)

Hypothesis:
- no-sort removes allocation/sorting, but makes nprobe small PQ calls;
- DiskANN PQ lookup is tiled/vectorized, so one fine batch may be faster.

Implementation:
- coarse top-k remains no-sort;
- selected child IDs are gathered into a Vec<u32> stored in
  DiskSearchScratch and reused through the object pool;
- one vectorized fine PQ call;
- no fine sort, canonical minimum only;
- no per-query heap allocation after scratch warm-up.

A/B: pooled-batch vs rung-2 no-sort. Pending.

## Prepared follow-ups

### Stage profiler
Instrumentation-only build reports:
- PQ preprocessing;
- coarse routing;
- fine routing;
- residual CPU/search.

Do not use profiler build for headline QPS.

### PQ gather locality
Semantics-preserving layout transformation:
- representative IDs and bucket boundaries unchanged;
- sort child database IDs ascending inside each bucket;
- state size unchanged;
- canonical winner unchanged.

A/B should require exact recall/I/O/hops/comparisons equality.

## After optimization converges

Rerun from scratch under the final path:
1. vocabulary size;
2. nlist;
3. nprobe;
4. memory/performance frontier;
5. realistic Recall@10 / L sweep;
6. properly recall-matched baselines.

Do not carry old CPU-derived knees forward automatically.
