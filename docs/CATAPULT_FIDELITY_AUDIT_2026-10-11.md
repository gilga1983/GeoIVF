# NavHints vs CatapultDB: fidelity audit (2026-10-11)

**Status:** source-level audit completed. Executable original-author oracle and same-SSD author-aligned sensitivity trials are running on self-hosted hardware; do not claim the runtime/IO deltas before their completion.

## Authoritative sources

- CatapultDB paper: https://arxiv.org/abs/2603.02164, Algorithm 2 (catapulted_lookup), §3.2.
- Original current author Rust source: https://github.com/MRandl/catapult-db/tree/a16eddd34b4339db5ec86e292470ce7929179bc3
- Original author starter and LRU: `src/search/hash_start/engine_starter.rs`, `src/sets/catapults/lru_set.rs`
- Original author LSH: `src/search/hash_start/hyperplane_hasher.rs`
- Original author search, including actual initial candidate handling: `src/search/adjacency_graph.rs`
- Our original controlled SSD adaptation, **not original author source**: `scripts/patch_diskann_paper_catapult.py`; Microsoft DiskANN3 pinned at `fcf90534174cf29c78c9f13b4cccf1fcabff85f5`.
- Our frozen held-out replay: `scripts/patch_diskann_catapult_snapshot.py`, `scripts/qualify_frozen_catapult_heldout.py`.

## What is faithful

CatapultDB's central mechanism is faithfully represented at the algorithmic level: Gaussian random-hyperplane LSH over the query; (2^H) buckets, each holding a small de-duplicated LRU of previous successful search destinations; base medoid and bucket contents as candidate start IDs; no modification of the underlying graph; insertion/refresh of the best search result into the query's bucket after search. Each bucket has an independent RwLock.

Our SSD trials use **H=8 and capacity 40**, consistent with the author's current `src/bin/run_queries.rs` default; 256×40×4 = **40,960 bytes of ID payload**, excluding hash planes and container overhead.

The paper says bucket eviction is LRU, and the original author's implementation indeed moves a re-accessed stored destination to the MRU end before evicting the oldest entry; the original port follows this.

## Discrepancies with author's released code

| Property | Original author | Our previous SSD adaptation | Judgment |
|---|---|---|---|
| Hyperplanes | StdRng + rand_distr StandardNormal, f32, plane-major | StdRng + manually implemented Box-Muller f32 | Same distributional family, not same seeded projections |
| Signature packing | Most-significant-plane first (left-shift) | Least-significant-plane first (`1<<h`) | Merely permutes bucket names with same planes |
| Medoid returned as answer | Admitted into its Catapult bucket | Skipped to avoid redundant starts | **Actual difference in LRU occupancy and later eviction** |
| Per-query timer | Search plus bucket feedback update | Prior port timer stopped before feedback insertion | **Latency slightly favors Catapult in earlier port** |
| Original-start distance | Full-precision vector L2 | DiskANN3 native PQ distance | Fundamental and explicit SSD-system adaptation |
| Initial candidate ordering | Buckets scored/sorted, medoid added; beam then bounded by search width | Medoid first; all bucket candidates scored by native DiskANN PQ; bounded L | May differ on ties/approximate ranks; not a byte-identical port |
| Causal learning | Author's native parallel CLI may interleave updates; pure algorithm observes previous completed queries | Some 4-thread frozen-training replays interleave updates | **Race-free but not strictly chronological**; 1-thread training in parity test |
| Catapult-use diagnostic | Author counts result ancestry to a catapult | Previous port counts nonempty bucket | Diagnostics are not equivalent (not an SSD-read metric) |

A subtle mistake in our original patch *comment*: it claims the author's code truncates initial candidate count to k. The actual Rust method is `Vec::shrink_to(k)`, which changes **capacity**, not **length**. Actual candidate-beam capacity is the search **beam width**. Our budget patch keeps DiskANN's configured L rather than silently widening it by the number of Catapult starts. This remains a controlled and reasonable design, but don't justify it by inaccurately claiming the author truncates to k.

## Scientific consequence

The previous Catapult-style comparator is not an invented weak baseline. It implements the authors' LSH-bucket routing, per-bucket LRU, and reuse of successful destinations on the **same PQ-compressed, SSD-resident index** that NavHints traverses. That controlled comparison is arguably better suited to isolating a routing mechanism than comparing author-native RAM search against our SSD system.

But we must not call it **bit-identical original CatapultDB**. The two material policy/timing discrepancies above require either correction or a measured ablation. Notably, both the medoid suppression and exclusion of update cost tend to *favor* the older adaptation, so the corrected baseline is not expected automatically to become stronger. Quantitative results, however, must be measured rather than inferred.

## Verification runs launched (self-hosted only)

1. [Native author-source oracle and compiled corrected DiskANN3 adapter](https://github.com/gilga1983/GeoIVF/actions/runs/38096585900), plus subsequent updated workflow dispatch. The native oracle invokes the authors' unchanged `EngineStarter` and checks deterministic signatures, 40-capacity LRU-like behavior, duplicate refresh and medoid residency against an independent equivalent model. It also checks the harmless bit-order permutation and the `shrink_to` misconception. The corrected SSD adapter is built using the original pinned DiskANN3 source. This is a **runtime parity test**, not an ANN accuracy claim.
2. [Same-SSD PubMed1M/MedRAG-Zipf source-fidelity sensitivity](https://github.com/gilga1983/GeoIVF/actions/runs/38096797188): compile original and author-aligned Catapult variants from the **same pinned DiskANN3 revision**, and replay the **same prebuilt 1M-vector index** and same query set. The first 5K queries train LRU states in strictly chronological one-thread order; the next 5K are evaluated with frozen LRU states on four threads. Three LSH seeds and L=20,40,80, with native Recall@10, logical SSD reads/query and measured latency. Only this run can establish whether the original mismatch materially changes our controlled comparison.

The corrected patch is optional and applied **after** the original adapter: `scripts/align_catapult_author_semantics.py`. Original baseline patches and the approved VLDB manuscript remain unchanged. Tests fail closed on source drift.

## Acceptance criteria

- Native author executable tests pass for all seeds and LRU events.
- Author-aligned SSD adapter compiles and uses exactly the same graph/PQ/SSD as old adapter.
- Validated 1-thread chronological training and 4-thread frozen evaluation with proper 5K GT, matched recall and multi-seed variation.
- Either report no material difference and retain old results with caveat, or replace table entries with properly labeled **CatapultDB (author-aligned SSD adaptation)** numbers.

Do not claim victory over author's complete system from this comparison alone. The original author RAM-only reproduction is a separate architectural validation.
