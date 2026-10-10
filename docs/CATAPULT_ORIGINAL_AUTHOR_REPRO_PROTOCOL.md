# Original CatapultDB code reproduction: locked protocol (October 11, 2026)

## Why this replaces Catapult-style as the literature-facing comparison

The current VLDB manuscript evaluates an *independently implemented Catapult-style routing mechanism* in the pinned DiskANN harness. That isolates mechanism and extra RAM, but is NOT the original authors' software. It must not be presented as the complete original CatapultDB implementation.

We located the authors' real public repositories:

- **Authors' actual Rust CatapultDB implementation:** https://github.com/MRandl/catapult-db, frozen `a16eddd34b4339db5ec86e292470ce7929179bc3`, updated September 22, 2026.
- **Paper experiment repository and submodules:** https://github.com/sacs-epfl/catapult-db-experiments, which links original `MRandl/catapult-db` and `sacs-epfl/catapulted-diskann`.
- **Authors' C++ DiskANN fork:** https://github.com/sacs-epfl/catapulted-diskann, frozen `c1dbaecce5e7e02d02ad1493660c8048d965005d`, February 13, 2026, the gitlinked revision from their experiments.
- **Research paper:** https://arxiv.org/abs/2603.02164, March 2026.

**Important systems difference:** The original authors' Rust code (`src/search/adjacency_graph.rs`) explicitly implements a **fully memory-resident graph**. Their C++ fork's CatapultStore is wired into memory-graph search (`src/index.cpp`), but the public fork's disk-search implementation (`src/pq_flash_index.cpp`) does not contain that Catapult integration. Therefore the original code cannot simply be executed against our disk-resident DiskANN graph as though it were an equivalent SSD path.

## Active experiments

1. **Original binary smoke:** workflow `.github/workflows/catapult-original-author-smoke.yml`; clone authors' original Rust code unmodified and execute both vanilla and real Catapult modes on its bundled tiny test graph. Self-hosted, hardware lock, archived author SHA, no external results claimed.
2. **Full native author-source Coveo:** workflow `.github/workflows/catapult-original-coveo.yml`; use authors' unmodified Rust search library, original authors' C++ DiskANN Vamana memory-graph builder, original chronological Coveo product/query embeddings. The benchmark harness is a *small external caller* of the unchanged author library, with no replacement of their LSH, catapult-store, beam search or eviction algorithm.

Full benchmark contract: **31,950 × 50 normalized product vectors**, padded with zeroes to 64 dimensions for the original Rust SIMD loader. Padding exactly preserves original Euclidean distances and cosine neighbor rankings. Original 5K disjoint prefix remains reserved as static-training traffic for NavHints; Catapult learns strictly causally on the next **20K** heldout production queries, and its measurements cover only the final **5K** queries, exactly as in the NavHints real-demand evaluation. Original Coveo cosine/inner-product exact ground truth IDs are retained.

Source graph: construct a Vamana memory graph using the authors' own C++ DiskANN fork at R48/L128/alpha1.2, serialize in its native graph/payload format, and load directly with the authors' Rust graph importer. Graph build runs once and is unchanged for both vanilla and Catapult. No graph-index reconstruction per method. Benchmark Rust engine operates with one sequential thread to maintain causal query order; compare both original-author modes at L=12/20/40/80/160, seeds 42/123/2026, rotated order. Catapult runtime configuration follows the author's current original `run_queries.rs`: 8 LSH hash bits and 40 ID slots per bucket. This is 40,960 bytes of ID payload plus projection vectors, containers and allocator overhead. Our NavHints Core is 102,940 bytes of auxiliary persistent routing payload; **not identical auxiliary RAM**.

Record native Recall@10, mean/p95/p99 latency, author counters for distance computations/visited nodes/catapult usage, seed-specific outputs. No numeric claim accepted until complete artifact validation.

## Scientific boundaries

This experiment is an **original-code reproduction on identical data**. It is **NOT** by itself a hardware- or memory-matched whole-system comparison of in-RAM Catapult vs disk-resident NavHints. Do not compare its memory-only throughput or 0 SSD reads against NavHints' SSD timing, and do not imply it implements authors' hypothetical disk integration. Keep the earlier controlled same-DiskANN Catapult-style comparison but label it exactly as an independent mechanism implementation. Put both evaluations, with their distinct aims and hardware resources, alongside each other only with explicit caveats.

When revising the paper, explain the separation: a controlled same-index routing-mechanism comparison (our port), and a reproducible native original-author system comparison (authors' code). **Leave all approved paper sections unchanged** until we have inspected native code results and chosen defensible common workloads and recall anchors.

Only `runs-on: [self-hosted, linux, x64, geoivf]`. The same `speed-device.lock` serializes all compilations and measurements against the Starling BigANN experiment. Licensed Coveo files never enter GitHub artifacts.
