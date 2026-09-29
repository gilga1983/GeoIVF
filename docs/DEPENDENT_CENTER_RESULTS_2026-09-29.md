# Dependent-center qualification results, 2026-09-29

Implementation commit: `241539bc59fe6a0aa3f6bceb023c2b45435c0f5f`.
Successful Actions run: `36526579015`; artifact ID: `11015290848`.
Artifact SHA256: `095c348dae39166f533d29e238c52d9b1775da7a52e988b5767413064160b965`.

## Scope

Canonical TexMex SIFT1M, all 1,000,000 original 128D FP32 vectors. Same frozen
4 KiB GeoPack pages, eight vectors/page, nlist=1024, nprobe=64, k=10, seed=12345,
and the same 64 development queries as the preceding PCA qualification. PCA
was fitted on the separate 100,000-vector learning file, not on these queries.
Only RAM summary encoding changes. All filtered encodings use 64-page planning
windows and zero-gap coalescing. The no-filter baseline uses whole-list planning.
These are requested page/extent counts from RAM payload replay, not physical
SSD traffic, latency or throughput. There is no CLIP comparison in this run.

Payload SHA256:
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA basis fingerprint:
`d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Candidate-route SHA256:
`16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.
All three fingerprints and the query IDs match the preceding projection run.

## Verified execution

The runner suite collected 82 tests: 76 passed, six skipped. All 22 new tests
passed. The six skips are optional io_uring tests because this workflow builds
the synchronous reader, not the optional liburing adapter; these skips do not
supersede the previous asynchronous-I/O validation. Native direct-I/O tests ran.

All 832 configuration-query comparisons matched the independent FP64 exhaustive
answer over the selected IVF lists. This is 64 distinct queries across thirteen
configurations, not 832 independent queries. All have Recall@10 = 0.99375.
96,503 rejected-page decisions were independently audited after search, with no
unsafe decision found. Audits cover the first two queries of each encoding and
include repeated pages; they are not distinct pages or independent queries.

The builder also decoded every representation and checked coverage of all one
million indexed points in each of the twelve encodings. Numerical guards and
outward-rounded radii are engineering safeguards, not formally verified
floating-point arithmetic. Online thresholds use only fetched vectors.

After artifact download, every CSV row count, query ID, aggregate mean, memory
component sum, read_bytes=4096*read_pages, and selected/gap/read-page relationship
was independently reconciled. PCA16/32/64 independent 8-bit variants exactly
reproduce the previous run's candidate/page/extent/stage/distance counts.

## Main comparison

Geometry includes actual packed coordinate bytes, four-byte radii, anchor-pair
IDs, interpolation coefficients, residual codes/scales, and byte padding. Add
eight bytes/page for the common radial interval; shared PCA/scales and other
directory arrays are additional. No expanded FP64 center table is retained.
The no-filter row's zero geometry denotes disabled filtering, not minimal-RAM
allocation in this diagnostic harness.

| Encoding | Geometry B/page | Pages/query | Aggregate page reduction | Median per-query reduction | Extents/query | Mean center error |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 8394.7969 | 0.0000% | 0.0000% | 65.0938 | - |
| pca16-independent8 | 160 | 6173.8750 | 26.4559% | 11.7122% | 595.9375 | 2.174 |
| pca32-independent8 | 288 | 2455.9219 | 70.7447% | 72.5192% | 822.5938 | 2.671 |
| pca64-independent8 | 544 | 297.9844 | 96.4504% | 97.0974% | 156.6406 | 3.086 |
| pca64-independent4 | 288 | 2189.4844 | 73.9186% | 76.5419% | 838.2031 | 52.459 |
| pca64-independent6 | 416 | 466.0625 | 94.4482% | 95.6050% | 249.9688 | 12.491 |
| pca64-midpoint2 | 166 | 8179.5625 | 2.5639% | 0.2460% | 224.9844 | 152.795 |
| pca64-midpoint4 | 292 | 7886.4219 | 6.0558% | 0.6602% | 311.8594 | 89.042 |
| pca64-midpoint6 | 418 | 7159.8750 | 14.7106% | 4.5149% | 583.0156 | 39.892 |
| pca64-interpolate4 | 296 | 7821.3594 | 6.8309% | 0.9706% | 337.2969 | 83.605 |
| pca64-interpolate6 | 420 | 6810.4531 | 18.8729% | 9.5959% | 826.9531 | 36.095 |
| pca64-interpolate4-res2 | 376 | 6020.5938 | 28.2818% | 15.4087% | 839.6562 | 60.905 |
| pca64-interpolate4-res4 | 440 | 498.2344 | 94.0650% | 95.2773% | 272.8281 | 10.658 |

Names midpointA/interpolateA use A explicit 8-bit PCA64 anchors and 8-A derived
centers. Independent4/6/8 denote bits per coordinate, not ball counts. Every
representation uses eight ball slots per page. Residual2/4 denotes correction
precision for the derived centers; two-bit residuals use three signed levels,
and four-bit residuals use fifteen. Full byte definitions and reconstruction
rules are in docs/DEPENDENT_CENTERS.md.

## Interpretation

Four explicit centers plus four midpoints reduce geometry from 544 to 292
bytes/page (46.32% less), but page reduction collapses from 96.45% to 6.06%.
The mean center error across all active ball slots rises from 3.086 to 89.042.
This average includes well-represented explicit anchors, so it understates error
in the derived centers. Allowing the best quantized interpolation coefficient
only lifts page reduction to 6.83% at 296 bytes/page.

Six explicit anchors help but do not fix the tradeoff: midpoint6 and
interpolate6 reject 14.71% and 18.87%, respectively, versus 94.45% for independent
six-bit centers at approximately the same geometry budget (416/418/420 B).

Adding four-bit residual corrections to the four-anchor interpolation recovers
94.06% page reduction at 440 B/page. However, independent six-bit centers use
416 B/page and request 466.06 pages/query instead of 498.23. They are both
smaller and more selective in this tested comparison. Two-bit corrections
are insufficient here (28.28% page reduction at 376 B/page).

Mean center error alone is not a sufficient predictor of page pruning. The
four-bit-residual variant has mean error 10.658, below independent six-bit
error 12.491, but its 95th percentile is 26.377 versus 13.898. A page is fetched
if any ball survives; concentration of error in the derived centers is one
plausible explanation for this result, not a separately proven causal claim.

A useful independent-quantization result: keeping PCA64 but lowering coordinate
precision to six bits saves 23.53% of geometry RAM versus eight bits while
retaining 94.45% aggregate page rejection. This is not equivalent I/O: 466.06
versus 297.98 pages/query is 56.41% MORE remaining page reads. It is a new
RAM/I/O operating point, not a domination of the richer representation.
At a 288-byte geometry budget, PCA64 four-bit centers request 2189.48 pages
versus PCA32 eight-bit centers' 2455.92, with no change in IVF answers.

Complete allocated directory arrays are about 74.58 MiB for PCA64 independent8,
59.26 MiB for independent6, 43.95 MiB for independent4, 44.43 MiB for midpoint4,
and 62.13 MiB for interpolate4-res4. These totals include the common structural
metadata and shared basis; Python/runtime and decoded query scratch are extra.
The new sidecar omits an unnecessary projected-centroid array used by the prior
head/tail implementation, so its total directory is about 1 MiB smaller at the
same geometry size. Comparison of geometry bytes is unaffected.

## Limits and next decision

This experiment tests within-page data-point anchors on the existing layout,
with one ball per point. Anchor subsets are exhaustively optimized for total
squared prediction error, not query false positives. Residual variants reuse
those interpolation-selected anchors. It does not test arbitrary fitted free
anchors, multi-point ball covers, cross-page dictionaries, or layout changes
intended specifically to produce interpolation structure. The negative result
therefore applies to this construction, not every possible dependence scheme.

The simple midpoint idea is safe and compact but not competitive on these pages.
Residual coding can recover selectivity; independent lower-bit centers remain
the simpler measured frontier in this sweep. Preserve the dependent variants as
ablations, and include 4/6/8-bit independent centers in subsequent real-I/O tests.
No speedup, held-out generalization, multiple-dataset result or CLIP superiority
is claimed. Prototype decoding and construction timings are archived but are
not optimized native performance evidence.

The complete artifact contains per-query CSVs, encoding breakdowns, error
statistics, query IDs/routes, learned PCA matrix, dataset provenance, test XML,
package versions and logs. GitHub retention is 30 days; retain the downloaded
archive for the research record.
