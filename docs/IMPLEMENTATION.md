# NavHints implementation contract

This document describes the **final design**, not its development chronology. Paper repository: [DemmandAwareANN](https://github.com/gilga1983/DemmandAwareANN). Source repository: [GeoIVF](https://github.com/gilga1983/GeoIVF).

## Search engine invariants

NavHints is a **routing extension** to a pinned disk-resident ANN graph, not a new graph index.

1. **Unchanged graph:** no online edges or graph construction updates, no new full-precision vectors in a separate RAM index.
2. **Unchanged search:** the engine's bounded best-first frontier, beam width, SSD record expansion and stopping rule are retained.
3. **Existing PQ:** every proposed vertex ID is scored for the current query using resident PQ codes / lookup table. Hint IDs alone do **not** imply returning a cached ANN answer.
4. **Causal history:** training requests are disjoint from measured queries; online destinations are introduced only after the originating query completes.
5. **Cache policy:** global unique FIFO, no duplicate age refresh. One top-1 successful result ID per completed query.
6. **Optional page exits:** the persistent layer, if enabled, adds up to ten junction-specific destination IDs to existing page slack and samples novel observations at the specified rate.

The principal allocation is **B=14,464 junction IDs**, **C=2,048 global destination IDs**, **k=8 additional starting candidates**. The directory has 512 coarse PQ groups and eight probed groups. Workloads with fewer qualifying training junctions, notably Coveo, retain fewer actual IDs instead of inventing them.

### Training

The offline collector records ordinary search traversals from **5,000 disjoint training queries** on the target index. For each visited vertex, its first expansion position contributes to a workload-weighted junction score. The top B eligible IDs are grouped into 512 geometric regions represented by medoids. The scored medoids and the children in eight query-nearest groups supply **one** junction entry. Only IDs and group offsets plus compact representative PQ codes are additional routing state.

Important code:

- \`scripts/patch_diskann_waypoint_trace.py\` and \`scripts/collect_navigation_trace_generic.py\`: read-only traversal tracing for public workloads.
- \`scripts/learn_global_navigation_landmarks.py\`, \`scripts/learn_navhints_generic.py\`: score and select eligible training junctions.
- \`scripts/build_hint_ivf.py\`, \`scripts/build_hint_ivf_generic.py\`: 512-region routing directory (PubMed and generic workload formats).
- \`scripts/patch_diskann_hint_ivf_packed_direct.py\`: optimized PQ-entry selection on pinned Microsoft DiskANN.

### Online RAM-only Core

At query start, select the best learned junction and PQ-score up to 2,048 unique recent successful result IDs. Maintain the ten best scored destinations using the existing PQ work, and offer **up to eight distinct destinations** to the original search frontier alongside the learned junction. Deduplicate IDs and respect the search frontier's existing capacity. The real engine decides whether to expand an offered vertex.

After completion, insert the top-1 result in the FIFO history if unseen, evicting the oldest inserted ID when full. This is not exact-answer caching, nor an LSH query-region map.

Important code:

- \`scripts/patch_diskann_experience_core.py\`: causal recent destinations and optional junction persistence.
- \`scripts/patch_navhints_regional_history.py\`: final global FIFO destination provider, explicitly configured with \`DISKANN_LOCAL_CACHE_MODE=fifo\` and \`DISKANN_LOCAL_RECENT_STARTS=8\`.
- \`scripts/qualify_final_core_public.py\` / \`scripts/qualify_final_core_coveo.py\`: four-arm online request replay and invariant checks.

**Do not infer a new storage-memory cache from the eight starting points.** All starts reuse PQ representations that the original DiskANN instance already keeps in RAM. The selected 102,940-byte PubMed figure accounts for ID/group/PQ-representative payload and excludes hash-table allocator, scratch and other process memory. See equal-payload evidence from [run #38120556463](https://github.com/gilga1983/GeoIVF/actions/runs/38120556463).

### Optional sampled persistence

Each selected junction may keep H=10 exit IDs co-located with an ordinary graph record when there is sufficient page slack. Sample1/2/4/10 admits a newly observed successful destination with probability 1, 1/2, 1/4 or 1/10. Accepted observations update one FIFO slot and use already known candidates to fill otherwise vacant slots; non-admitted requests need no page update. Returning through these exits never bypasses the original ANN search.

The [selected-Core PubMed write sweep](https://github.com/gilga1983/GeoIVF/actions/runs/38137546048) finds that RAM-only Core already captures almost the entire gain on this workload. Writes/query refer to **logical page-update requests** at a fixed L, not physical NAND writes, write amplification or a proven crash-recoverable SSD transaction.

### External methods

**CatapultDB:** the author source uses Gaussian random hyperplanes and per-bucket LRU winner histories. The adapter checks source fidelity and executes its routing mechanism in the same pinned **DiskANN SSD engine**. It is not a comparison against the author's original complete in-memory graph engine. Code: \`patch_diskann_catapult_snapshot.py\`, \`align_catapult_author_semantics.py\`, \`patch_catapult_author_causal_eval.py\`, \`qualify_optimal_navhints_catapult.py\` and \`audit_catapult_source_parity.py\`.

**Starling:** the original \`zilliztech/starling\` index, graph-page layout, native search algorithm, memory navigator and SSD page accounting are retained. The final-core patch adds only PQ-scored routing starts and causal history; it also trains a directory from Starling's **own disjoint training traversals**. Code: \`patch_starling_final_core.py\`, \`generate_starling_final_core.py\`, \`summarize_starling_final_core.py\`; their historical source-patching prerequisites are preserved. Full native results on Coveo and BigANN are recorded separately; BigANN's native gains are close to zero.

**Caching/DiskANN++ controls:** \`prepare_diskannpp_qsev.py\`, \`patch_diskann_qsev.py\`, \`learn_hot_cache_ids.py\`, \`patch_diskann_hot_cache.py\` and the frozen competitor harness isolate specific navigation or record-caching mechanisms. They do not claim to reproduce the complete third-party indexing system.

## Data flow

\`\`\`text
5K disjoint training requests
    -> ordinary graph traversal traces
    -> ranked junction IDs
    -> 512 PQ medoids + bucketed junction-ID directory

Query q:
    resident PQ LUT
    -> one trained junction + <=8 PQ-scored recent successful IDs
    -> native best-first frontier -> ordinary SSD graph reads
    -> optional W(h) exits (from already requested junction page)
    -> fresh search result
    -> unique FIFO update, and optional sampled W(h) page update

Measured suffix:
    4K causal warmup / 1K measured (PubMed, Text2Image, BigANN)
    20K causal warmup / 5K measured (chronological Coveo)
\`\`\`

All author-engine changes are made to isolated, pinned external checkouts by patch scripts and archived as diffs. No executable or output uses another experimental prototype silently. For commands, data prerequisites and artifact names see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
