# Exact-decision Starling NavHints fast selector: frozen Coveo protocol

## Research question

Does the original Starling port make NavHints look unnecessarily expensive by using scattered, 16-ID PQ batches and a full sort for each of 512 region candidates? The high-locality Coveo study demonstrates RAM-only Core improving original Starling at L=20 from 23.592 to 19.782 logical reads/q (16.15%) and from 479.7 µs to 450.1 µs (~6.17% latency). Before interpreting the 35 µs selection overhead as fundamental, measure an optimized implementation at **identical routing decisions**.

This experiment **does not propose a new NavHints algorithm**. The paper's approved algorithm, 16K directory, Recent512 FIFO and search policies stay unchanged.

## Scorer modifications, both measured under the same binary

Classic `core16k_recent512`: current original-Starling adapter. Scores Recent512 in 16-ID PQ batches; scores 512 region medoids in 16-ID batches, repeatedly sorting the eight best regions; scores each selected region's children in 16-ID batches; injects the lowest PQ-distance eligible Recent512 ID plus the lowest PQ-distance eligible junction.

`fast_core`: invokes the same native `pq_flash_index_utils::pq_dist_lookup` with the same resident PQ codes, same chunk order, same distances and IDs, same `visited` exclusion, same top eight cells and candidate tie-breaking. Optimizes only memory access/scratch use:
- Query-local packed PQ codes to score the 512 recent IDs in **one batch**.
- Query-local packed PQ codes to score the 512 region medoids in **one batch**.
- Exact `lower_bound` insertion into the short top-eight cell list, instead of sorting the entire list 512 times.
- One packed PQ pass for the members of **each selected region**, while processing the regions in their original order and keeping the original strict distance comparison.
- Two query-local reusable, resizable vectors (packed codes and scores). No persistent vector-cache or graph change; incremental temporary query memory bounded by 512×PQ-chunks bytes plus 512 floats and allocator overhead, included in timed execution.

`recent512` and `fast_recent512` isolate the dynamic-cache scorer.

No DIM, metric, original Starling in-memory navigator, SSD graph layout, routing candidate set, training, results, region table, PQ encoding, or stopping rule is changed.

## Dataset and replay

Original and fast modes use the *same compiled binary and index*, pinned original Starling commit `17dc3e8a011533a62374445f53963e951b72883a`, and frozen real Coveo SIGIR eCom 2021 catalog (31,950 × 50 L2-normalized product embeddings). Native L2² preserves cosine ranking. Disjoint chronological 5K static training, 20K online warmup, 5K measured suffix, three repetitions, 4 search widths L=12/20/40/80, one SSD lock, rotated run order. Same original-source patch and diagnostics instrumentation is present in both arms. Runs only on `[self-hosted,linux,x64,geoivf]`.

The workflow `.github/workflows/starling-coveo-fast-core.yml` saves native I/O, latency, recall, first eight SSD page IDs and per-query candidate IDs + PQ scores + pop/read counts for every mode/rep/L.

## Hard correctness gates

For **each** of 60,000 Core vs fast Core paired measured queries (3×4×5K), verify selected recent ID, learned junction ID, their PQ distances/ranks, insertion, expansion, page-read flags, total logical I/Os, first eight physical page IDs, page count, cache hits, and original Starling Recall@10. Also verify the analogous **60,000** recent-only vs fast-recent pairs. Any mismatch causes an explicit failure. Only if all **120,000 matched pairs** validate will the workflow print performance results.

Compare within identical widths (hence exactly matched recall in successful pairs) using native per-query mean latency. The two key metrics are `recent_ns + junction_ns` (selection work) and `stats.total_us` (end-to-end query cost). Also report native logical I/Os vs unchanged Starling `baseline`. Avoid interpreting query replay QPS as 4-thread native throughput, or interpreting logical I/O operations as externally audited block-device activity.

Do not modify approved manuscript text or copy licensed Coveo source vectors to the repository.
