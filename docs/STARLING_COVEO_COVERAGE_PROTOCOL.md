# Original Starling on real chronological Coveo: complementary entry test

**Status:** Experimental branch, not yet manuscript evidence. This protocol is fixed before results are interpreted.

## Why this workload

Coveo SIGIR eCom 2021 is the strongest demand-locality workload in the existing NavHints manuscript: **Full Sample2 versus DiskANN reduces SSD reads 62.25% and latency 70.47% at matched recall**. Those numbers are from *DiskANN with persistent hints*; **this experiment is RAM-only Core on original Starling, so those gains are not transferrable**.

Reuse the previously prepared, licensed, timestamp-sorted frozen Coveo split from the *successful* `experiment/real-demand-coveo` run:
- Product base: **31,950 × 50 float32** embeddings, nonzero and L2 normalized.
- Static training: chronological queries **0–4,999** (5,000), **disjoint** from online evaluation.
- Causal online learning and data-cache warming: queries **5,000–24,999** (20,000).
- Measured evaluations: **25,000–29,999** (5,000).
- Ground truth: exact inner-product/cosine neighbors already computed on the whole catalog and sliced to the same 25,000 replay queries. Native Starling's squared-L2 ranking on normalized data is equivalent to cosine ranking; do not claim numeric score equivalence.

No downloader, license bypass, or new pseudo-random locality is used. The native Starling build consumes existing local Coveo `base.fbin` and the original timestamp order; previous parser/manifests are hash-checked, and no full dataset copies are attached to the Actions artifact.

## Original source and controlled changes

Pinned original `zilliztech/starling@17dc3e8a011533a62374445f53963e951b72883a`; its own float/L2 memory navigator, graph build, page partitioning and search remain unchanged. The temporary native source patch only exposes an extra RAM-only candidate selection interface and precise diagnostic timing, and the same patch machinery is used for every arm.

On the `experiment/starling-coveo-coverage` branch, see:
- `scripts/generate_starling_coveo_coverage.py`: same author-index builder, shared SSD lock, source-provenance preflight and time-order causal replay.
- `scripts/patch_starling_coveo_coverage.py`: 20K/5K replay and alternative routing objective.
- `scripts/summarize_starling_coveo_coverage.py`: strict aligned per-query I/O/latency, coverage, recall, and ID-change analysis.
- `.github/workflows/starling-coveo-coverage.yml`: **self-hosted only** and resource-limited.

## Seven arms, three repetitions

- `baseline`: unmodified native Starling routing (instrumented).
- `recent512`: only chronological FIFO of prior 512 *completed* top-one query results (causal).
- `core16k_recent512`: nearest-PQ 16K Starling-trained junctions, grouped in 512 regions, plus Recent512.
- `diverse_core`: previous weak 0.35×native-best-distance filter (control).
- `maxmin_core`: within the 24 nearest candidate hints and **within 2× the best hint query-distance**, select the candidate *most separated from its nearest original Starling starter* (PQ approximation), excluding shared 4KiB pages.
- `cover_core`: same shortlist, page exclusion and 2× relevance bound; maximize minimum PQ separation from Starling starters *per unit of PQ query distance*.
- `gate50_cover`: coverage selection only when native Starling's best candidate distance exceeds the **median of preceding 20K warmup native-entry distances**. No oracle or labeled tuning.

All arms retain the same original Starling memory navigator (`MEM_L=10`), no extra persistent vectors or SSD hints. The 16K hint ID state, 512 recently successful destination IDs, and 24-element transient shortlist are accounted for separately from Starling's much larger in-memory navigator. All three repetitions are rotated to mitigate hardware-order effects.

Four search widths `L∈{12,20,40,80}`, beam width 8, Recall@10, full original 31,950-product index. Smallest L=12 is above the 10 original Starling starter slots and leaves space for two additional hints; do not accidentally test `L=10`, which leaves no room for the added entries.

## Interpretation and validation

Native reported I/Os mean **logical Starling page-read operations**, not independently counted physical NVMe reads. Include native latency with candidate scoring in measured query time. Compare all methods at the **same recall**, using only genuine overlapping recall ranges. Same-L comparisons are diagnostic only if recall differs materially. Do not conflate RAM-only Core, Full Sample2, independent Catapult/QSEV mechanism implementations, and original author-system Starling.

The decisive diagnostics are (a) how often coverage chooses a **different** hint than nearest-query Core, (b) whether the first physical disk pages genuinely change, (c) the resulting SSD page count and latency, and (d) the fraction of weak-native queries that justify activation.

**Do not edit approved manuscript paragraphs** until these findings have been validated and approved.
