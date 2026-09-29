# Fitted-line and line-first packing results, 2026-09-29

## Executed experiments and provenance

Free-line/shell/adjacent-refinement implementation:
`e3c8e51253c5ead01bcec974afa882009d297748`.
Run `36530008273`, artifact `11016481981`, SHA256
`37835e5f43d7305a806b668f555edf8536a45b58321e50306dd6f3a1401eaabb`.

Broader line-first packing implementation:
`0b1a9f4590fe4b356c6e867f4f0fa8e409870cc8`.
Run `36530719490`, artifact `11016527866`, SHA256
`8b5643e1f671465049f100bde237ee2591af6aeb8337a6e672f603215d3ad1da`.
Both workflows completed successfully on the user's GeoIVF runners.

Both use canonical SIFT1M, all one million unchanged 128D FP32 vectors, eight
vectors per 4 KiB page, nlist=1024, nprobe=64, k=10, construction seed 12345,
and the SAME 64 development queries. Global PCA is fitted on the separate
100000-vector learning file. Queries do not fit the lines or page layouts.
Every method receives identical preassigned IVF candidate lists. All results
have Recall@10=0.99375 and match the candidate-set exhaustive answer.

Source GeoPack payload:
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA basis:
`d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Routes:
`16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.
All match prior canonical representation studies. Original-layout independent
baselines repeat their prior requested-page/extents/stage/distance counts exactly.

This is RAM replay of actual FP32 payloads, not measured SSD traffic, latency,
throughput, or an upstream CLIP comparison. No held-out or multiple-dataset claim.

## What was implemented

Fit a free least-squares affine line to each page in projected space. Store a
quantized origin and direction, then recompute along-line coordinate t and
perpendicular norm rho relative to the DECODED normalized line. Store enclosing
scalar bins rather than retaining residual directions. The safe ideal bound is
hypot(distance(t_query,T_point),distance(rho_query,R_point)). Taking the minimum
over page points yields a conservative page bound. Combined mode additionally
uses the existing deterministic radial interval, not CLIP's learned bound.

The matching ball ablation encloses each scalar-defined region in an ordinary
ball centered on the line. It discards orthogonality information; in exact
arithmetic its lower bound is no stronger than the scalar-shell bound.

Default 64D line geometry costs 172 B/page: 64 B origin, 64 B direction,
32 B uint16 scalar codes for eight points, and 12 B FP32 scalar quantization
parameters. The common radial interval adds 8 B/page. Shared PCA, coordinate
quantizer metadata, IDs, valid counts, list ranges and IVF centers are additional.
No expanded axes are retained at load time. FP32 line axes cost 556 geometry
B/page; the full 128D quantized line costs 300 B/page. The 64D line's complete
allocated directory arrays occupy 30.07 MiB, excluding runtime and query scratch.

We tested three layouts, not merely three encodings of the same pages:

- Original: the preceding frozen 16-coordinate GeoPack layout.
- Adjacent-line refinement: four alternating passes of capacity-preserving
  reassignment between adjacent full pages inside each IVF list.
- Line-first pool64: regroup within blocks of up to 64 vectors inside an IVF
  list. Evaluate 251 fixed candidate endpoint pairs, greedily choose eight-point
  groups by line residual/axial-span score, and freely refit their final lines.
  Keep a proposed block only when its refitted objective beats the original.

Both alternative layouts start independently from the original GeoPack layout.
Neither crosses IVF lists, changes page count/capacity, modifies FP32 coordinate
values, or uses evaluation queries. Partial tail pages remain fixed.

The refitted surrogate is sum(rho^2)+max(rho^2)+0.1*axial_span^2 per page. The
pairwise and greedy-pool searches are heuristics, not global packing optima.
All filtered searches use 64-page windows and zero-gap coalescing. The unfiltered
baseline allows contiguous whole-list planning; it requests 8394.796875 pages
and 65.09375 extents/query on this query set.

## Main comparison

Geometry below excludes the common radial interval and shared/structural arrays.
A geometry budget of 176 B admits both the 172 B line and 176 B PCA18 control;
these are close-budget, not byte-identical, encodings. All methods have the same
IVF answers, and requested pages/bytes are not physical-device measurements.

| Layout | Representation | Geometry B/page | Pages/query | Page reduction | Median reduction | Extents/query |
|---|---|---:|---:|---:|---:|---:|
| Original | Line64, ball bound | 172 | 7817.3438 | 6.8787% | 0.4623% | 313.1094 |
| Original | Line64, shell bound | 172 | 7754.8438 | 7.6232% | 0.5245% | 328.0938 |
| Adjacent refinement | Line64, shell bound | 172 | 7710.4063 | 8.1526% | 0.7012% | 347.4375 |
| Line-first pool64 | Line64, ball bound | 172 | 7479.0469 | 10.9085% | 2.3694% | 444.9531 |
| Line-first pool64 | Line64, shell bound | 172 | 7418.5625 | 11.6290% | 2.5854% | 459.4844 |
| Original | Independent PCA18, 8-bit | 176 | 5732.0625 | 31.7189% | 18.5084% | 679.9375 |
| Adjacent refinement | Independent PCA18, 8-bit | 176 | 5706.3125 | 32.0256% | 19.0257% | 697.6563 |
| Line-first pool64 | Independent PCA18, 8-bit | 176 | 5646.0781 | 32.7431% | 20.1215% | 757.9531 |
| Original | Independent PCA64, 4-bit | 288 | 2189.4844 | 73.9186% | 76.5419% | 838.2031 |
| Line-first pool64 | Independent PCA64, 4-bit | 288 | 2102.2813 | 74.9573% | 77.2845% | 899.1406 |

Broader line-oriented grouping increases scalar-line page elimination from
7.62% to 11.63%, but does not close the gap to independent PCA18 at nearly the
same geometry RAM. Its median query saves only 2.59% of pages. Every query does
save something, but the largest query contributes 12.84% of total saved pages.

Independent controls benefit from the repacking too: PCA18 improves from 31.72%
to 32.74%, and PCA64 four-bit from 73.92% to 74.96%. The layout's small benefit is
not exclusive to implicit centers. The better byte pruning of independent codes
comes with more extents in the close-budget comparison (758 versus 459), so this
is NOT an end-to-end speed ranking or dominance on every resource metric.

## Why the line model remains weak

| Layout | Mean variance explained by best fitted PCA64 line | Median explained | Mean perpendicular distance |
|---|---:|---:|---:|
| Original | 30.9831% | 29.7639% | 140.1514 |
| Adjacent refinement | 32.1844% | 30.7501% | 138.0351 |
| Line-first pool64 | 36.8431% | 34.2468% | 130.7826 |

The broader builder really changed the grouping: 15726 of 16035 pools accepted
new membership, and 123295 of 125445 pages (98.286%) changed their vector sets.
The total fitting objective fell 9.662%. Nevertheless, most within-page variation
remains outside the fitted line. The scalar description discards the directions
of those substantial residuals.

Increasing axis precision does not fix it. On original pages, quantized line64
reads 7754.84375 pages/query, versus 7755.109375 for FP32 axes. Removing global
projection truncation also does not fix it: quantized line128 reads 7769.640625
pages (7.4469% reduction). These diagnostics support residual angular information
loss, not ordinary axis quantization error, as an important limitation in this
construction. This is an interpretation, not an impossibility theorem.

The mathematical two-scalar lower bound remains valid. The practical premise
that these full pages can be made sufficiently thin was not met by either
packing heuristic tested here. This does not rule out whole-list global grouping,
other workloads, multiple local basis directions, or shared model dictionaries.
Curves were not tested. More flexible geometry also has a storage/decoding cost.

## Correctness and independent verification

First workflow: 95 tests collected, 89 passed, 6 optional io_uring skips. All
13 new line tests passed. Follow-up: 96 collected, 90 passed, the same 6 optional
skips, including the added pool-layout test. Existing prior asynchronous-I/O
validation is unchanged; these workflows intentionally build the synchronous
reader and use RAM replay for their algorithmic experiments.

First experiment: 1600 configuration-query matches over 25 settings; 155792
rejection decisions audited. Follow-up: 640 matches over 10 executed settings;
43955 decisions audited. Every comparison and audit passed. Both experiments
use the same 64 distinct development queries; five original-layout settings are
repeated in the second experiment. The counts must not be advertised as thousands
of independent queries or distinct pages.

Line scalar intervals were checked against every indexed vector in each built
encoding. Independent controls retain their all-point covering audit. After each
packing step every IVF list's multiset of vector IDs was checked. Rejection audits
read original vectors only AFTER the online search and do not seed its threshold.
All reported methods match independent FP64 candidate-set search with ID ties.
Numerical safety uses engineering guards, not a formally verified arithmetic
library.

Both downloaded ZIP digests match GitHub's artifact digests. CSV query IDs,
all aggregate means, read_bytes=4096*read_pages, selected/gap/read counts,
geometry component sums, route fingerprints, original baseline repetitions,
and shell-versus-ball page counts were independently reconciled. The first
experiment's five original-layout independent baselines also reproduce the
preceding dependent-center study exactly.

## Decision

Retain these line implementations and both packing algorithms as reproducible
ablations. Independent quantized centers remain the leading page-pruning
representation for the main evaluation. There is no demonstrated SSD speedup,
held-out generalization, or CLIP superiority from these experiments.
Reproduction: scripts/qualify_lines.py and scripts/qualify_linepool.py; details
of the scalar encoding and proof are in docs/LINE_QUALIFICATION.md. Preserve
the downloaded artifacts beyond GitHub's 30-day retention window.
