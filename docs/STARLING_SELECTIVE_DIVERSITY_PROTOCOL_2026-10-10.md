# Starling NavHints: selective and diverse native entries (2026-10-10)

## Motivation

The original-Starling three-repetition diagnostic [run 38075966365](https://github.com/gilga1983/GeoIVF/actions/runs/38075966365) verified that:
- Score-only Core preserved Starling's native disk-page path but added ~96 µs/query at L=20.
- RAM-only Core saved ~0.237 native logical I/Os/query at L=20, while a **ground-truth oracle diagnostic** saved ~2.968.
- Recent512's selected candidate beat Starling's best initial candidate on only ~0.8% of queries; learned16K's on ~10.9%. The originally chosen candidates often overlap the areas represented by Starling's native starts.

A shared source of redundant routing is that native Starling already returns **10** PQ-scored starting vertices from its in-memory navigator, whereas original NavHints' 16K candidate selector minimized only PQ(query,hint) and excluded duplicate IDs but not nearby entries.

## Six-arm test (predeclared, not suffix-tuned)

**Repository branch:** `experiment/starling-selective-diverse`.

**Workflow:** `.github/workflows/starling-selective-diverse.yml`, strictly `runs-on: [self-hosted, linux, x64, geoivf]`.

**Original pinned Starling:** `zilliztech/starling@17dc3e8a011533a62374445f53963e951b72883a`, same original author graph, disk layout, memory navigator and stopping rule. Temporary source patch only changes optional hint selection plus diagnostics. No published Starling binary modified.

**Data:** BigANN-10M uint8/L2 and original GT, 5K disjoint queries to train Starling-native top16K junctions into a 512-region ID-only Hint-IVF. From a separate frozen 5K heldout set, first 4K causal online/warmup, last 1K measured, three repetitions in permuted order at L=20/40/80, native beam width 8.

**Arms:**
- `baseline`: original Starling, no NavHints entries.
- `core16k_recent512`: original Starling + RAM-only 16K junctions + causal Recent512, always active, originally nearest query-scored candidate.
- `diverse_core`: same resources and scoring but choose an entry geometrically distinct from Starling's ten native starts.
- `gate50_core`: original Core candidate selection but only activate for queries where Starling's native best-entry PQ distance is in the weakest ~50%, threshold computed from past 4K warmup queries.
- `gate50_diverse`: previous two mechanisms combined.
- `gate25_diverse`: diverse with a stricter ~25%-activation gate (75th percentile of native best query distance).

These are **diagnostic thresholds**, not optimally tuned parameters. No ground truth or validation performance is used for gating or selecting entries.

## Diversity definition

All starting vectors have resident Starling PQ codes; the hint candidates are already scored by PQ(query,candidate). Restrict each recent/junction source to **24 query-closest candidates**, then consider candidates in that order. Reject any whose **4 KiB graph-page ID** matches a native starter page, or whose minimum **symmetric reconstructed-PQ squared L2 distance** to the 10 native starts is less than **0.35 × PQ(query,best_native_entry) squared L2**. If no candidate qualifies, add none. This tests genuinely alternative directions rather than inserting more of the same.

This requires **no extra disk reads** and no persistent vector metadata, but does perform PQ reconstruction and pairwise calculations for the short list. Transient per-query scratch includes ~10×128 floats and two 24-item candidate buffers; record runtime costs. The native Starling memory navigator remains present and its (larger) memory must not be attributed to NavHints.

## Experimental validation

Archive source patch, all run logs, 3-repetition native Recall@10/mean-IO/latency tables, and **54,000** measured query-arm-width records including native start quality, gate openings, hint originality, insertion, expansions, page paths and scoring time. Verify actual query alignment and that gated-off queries neither score nor inject candidates. Compare both equal-width and recall-matched performance; do not interpret small recall shifts as equal recall without qualification.

**Do not touch the approved VLDB manuscript until results are assessed.** Distinguish our independently reimplemented QSEV and Catapult mechanisms from original native Starling; this test *does* modify a temporary checkout of original Starling for an optional routing interface.
