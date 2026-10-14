# Native LAANN + NavHints: RQ4 portability experiment

**Status:** Experimental branch, **not paper evidence** until a native real-SSD run completes, exact recall support is audited, and the full compiled engine is validated. Do not modify RQ4 or cite numerical results from a queued/running/failed workflow.

- Original authors' repository: [Dingyi-Kang/PageANN](https://github.com/Dingyi-Kang/PageANN), which contains LAANN as a distinct search mode.
- Pinned native source: `58fe40feb58d0ca1a168816b36c4d4b22d3eb29c`.
- Adapter: `scripts/laann_navhints/patch_laann.py`, adding `nh_laann_state.h` to native `src/pq_flash_index_laann.cpp`. PageANN search is not used.
- Run: [.github/workflows/navhints-laann-native.yml](../.github/workflows/navhints-laann-native.yml), self-hosted Legion runner only; 2 build threads, one in-flight query for online causality, serialized SSD.
- Baseline: original **LAANN** with its native navigation graph, page-aligned SSD index, 20% cached pages, full PQ codes in RAM, look-ahead and I/O pipeline.
- Comparison: baseline, junction-only, and junction+recent destination starts in the **same pinned LAANN engine**. Native reordered IDs, graph, PQ codes, page cache, navigation algorithm and pipeline stay unchanged. The adapter adds candidates to the native frontier after in-memory nav, without changing page graph records.
- Port-specific junction opportunity: learned from *page expansions*, mapping each visited SSD page to its first reordered vector ID. Unlike DiskANN's vector-level traversal, LAANN's training evidence is inherently at page granularity. The junction scoring is the sum of first-expansion positions across disjoint training queries, tie-broken by support and ID. A deterministic five-round PQ-reconstruction-space clustering forms 512 geometric regions from at most 14,464 learned page representatives. The selected eight regions are probed, then one junction ID is offered.
- Recent destinations: up to 2,048 unique, strictly causal first-result vector IDs in a global insertion-order FIFO (duplicates do not refresh). Up to eight nearest PQ-scored historical IDs are offered, with at most one per SSD page. These are **navigation seeds**, not cached answers.
- PQ scoring: uses the native already-resident `_cached_pq_buff` and query lookup table. No extra dataset-wide PQ copy and no additional SSD fetches for scoring IDs. Extra region metadata, IDs, history and temporary PQ-scoring buffers must be accounted for when comparing RAM.
- Evaluation: canonical SIFT1M is a first **integration/portability pilot**, not one of the paper's four workload measurements. The upstream authors' build scripts produce the Vamana graph, LAANN page graph, in-memory navigator, reordered PQ file, and **reordered ground truth**. Never evaluate against original SIFT ground-truth IDs.
- Split: first 5,000 queries train junctions; last 5,000 are replayed sequentially, with 4,000 online history warm-up and only the final 1,000 measured. Each baseline/junction/core arm at each L runs as a **fresh OS process**, avoiding history contamination across search widths.
- Measurements: native LAANN SSD reads, feedback-inclusive query duration and Recall@10 for the measured suffix. Use only jointly reachable, *measured* NavHints recall targets, with interpolation and no extrapolation. No between-engine latency ranking. Same deterministic query stream is reused, so repeated L sweeps are **not independent workload samples**.
- Scientific boundaries: This initial pilot measures one request at a time. It does not establish throughput under concurrent updates or generalize SIFT locality to Coveo, PubMed, Text2Image, or BigANN. It does not train an adaptive activation policy. A 1M/10M follow-up on matched paper workloads is necessary before integrating a third-engine conclusion into RQ4.

## Implementation and reproduction

The `qualify_native.sh` script builds the unmodified authors' source plus an isolated adapter; there is no external replacement ANN search harness. It reuses the existing validated GeoIVF SIFT1M cache or acquires the canonical archive through `scripts/fetch_sift1m.py`, and uses the upstream `sift1M/build_laann_sift1m.sh` with only build-thread count reduced. Index artifacts are cached on the self-hosted Legion and are not committed.

Run on an authorized, configured self-hosted machine through the GitHub Actions workflow. The build requires LAANN's original CMake dependencies (Boost, MKL, libaio, etc.); the workflow does not install system packages automatically. Generated artifacts include upstream SHA, patch, compile logs, native page trace, learned region directory, per-query CSVs, process-memory/time logs, and matched-recall JSON. A compile failure or absent matched-recall overlap is a failed gate, **not** a positive result.

The active manuscript [RQ4](https://github.com/gilga1983/DemmandAwareANN/blob/main/sections/04-evaluation-starling.tex) is deliberately left unchanged until successful results can be reviewed.

## Validated SIFT1M pilot results (October 14, 2026)

[Successful native-LAANN GitHub Actions run #38514613418](https://github.com/gilga1983/GeoIVF/actions/runs/38514613418), workflow artifact ID **11872010730** (`native-laann-navhints-20261014`), source SHA `1d4ed97f0d1b2baaade2a88c76f19d4e431baf20`. `status.txt` contains `NH_LAANN_REAL_SSD_PASS`; all 21 independent (7 widths by 3 methods) per-query CSVs and native indexing/training logs are present. The source patch preserves LAANN's native page graph, in-memory PQ array, sampled-page navigator, native cache and look-ahead/I/O pipeline.

**Correction to the archived analysis JSON:** the successful run's initial summary counts seven anchors because two search widths give *the same* 99.84% Recall@10. This duplicates a recall target and slightly biases its descriptive mean. The later `scripts/laann_navhints/derive_results.py` fix at commit `1bbe5ed0769bebfd49494b65bb22b6484ee42598` deduplicates recall targets before averaging. Do not quote the old seven-anchor summary without this correction.

After deduplicating the six jointly reachable Core-achieved targets (98.93, 99.43, 99.63, 99.80, 99.84, 99.88% Recall@10), averaging independent percentage differences by *recall target*:

| Native LAANN metric | LAANN+NavHints versus baseline | LAANN+NavHints versus junction-only |
| --- | ---: | ---: |
| Mean matched-recall logical SSD page reads | **0.94% fewer** | **0.027% fewer** |
| Mean matched-recall complete-request latency | **28.35% slower** | **10.84% slower** |

The native baseline already achieves 98.92% Recall@10 at L=20 on SIFT1M. At the highest jointly reachable 99.88% target, LAANN+NavHints actually requires **16.19% more logical page reads** after recall matching, although other anchors show positive read savings. These are descriptive measurements from one deterministic request population, not a robustness or statistical significance result.

**Where the overhead occurs:** the authors' own measured navigation-phase timer rises by approximately **195.8 μs per query** on average across the seven fixed-L tests. About 54.9 μs comes from scoring learned junctions, and a further 140.9 μs from the global 2,048-ID PQ-history scan and candidate insertion. This diagnostic is from equal-L native timing fields, **not** another matched-recall percentage. Additional per-request latency includes native search-path variation and SSD effects.

**Interpretation:** native LAANN integration and causal measurement are validated; the SIFT1M performance case is weak. The page-level in-memory nav graph is already effective at the tested high recall and leaves little room for historical destination seeds. The combined configuration is **not appropriate as a positive RQ4 performance claim**. A meaningful further paper evaluation should use a demand-structured dataset (e.g. PubMed or Coveo) and/or a stronger low-to-mid recall range while retaining both the native navigator and fully accounted PQ-scoring overhead. Do not tune only the NavHints arm or compare different native LAANN index builds across arms.

The paper repository and current RQ4 text remain unchanged.
