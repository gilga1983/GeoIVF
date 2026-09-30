# Equal-budget mixed-bit results, 2026-09-30

## Provenance and protocol

Tested implementation: `852a0a9ca697557bc38d70f930fd871fa021d8f1`.
Successful workflow: `36693316926`; job: `109815327071`.
Runner: `geoivf-gilga-Legion-Pro-7-16AFR10H-06`.
Artifact: `11088055808`, `mixedbits-36693316926`.
Downloaded ZIP SHA256:
`b1de5afaa6bb041682ac1294b35e0362d850f2842c68d083be1c1cf068d0d8ef`.

This is a completed canonical SIFT1M experiment. All 1,000,000 original 128D
FP32 vectors, 125445 4 KiB pages, within-list GeoPack membership/order, IVF
assignments, nlist=1024, nprobe=64, k=10 and construction seed=12345 are fixed.
The unchanged payload hash is
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
No geometry or payload regrouping is credited to the allocation change.

All eight code families use 512 coordinate-code bits/vector plus an outward
FP32 measured-error radius: 544 geometry bytes/full page. Shared allocation
plans, matrices, quantizers and PQ codebooks are additional and accounted for.
All mixed variants retain ALL 128 PCA dimensions with widths in {2,4,6,8}.
Zero-bit omission, odd bit widths, per-list quantizers and clipping optimization
are outside this experiment. Fixed global allocations keep vector records at
64 bytes; no per-vector variable-length code or precision header is introduced.

PCA and allocation variance/error statistics use the separate 100000-vector
learning file. Scalar min/max bins use indexed-base extrema, matching the old
uniform controls. Plain PQ uses the same 50000 learning vectors and 20-iteration
training policy as the preceding experiment. No evaluation query trains codes.

Development: permutation(seed=20260929)[:128], 82 arms, three randomized
interleaved direct-I/O rounds. Sweep shortlist R over16/32/64/128/256 for each
representation and approximate/certified mode, choosing the fastest setting
meeting >=99% strict global Recall@10. Freeze those choices before test access.
Held-out: permutation[1768:2024], 256 NEW queries excluding every previous
cohort. Twenty-two selected/predeclared arms run three direct-I/O and three
RAM-replay rounds. Fixed-R64 approximate comparisons were specified in advance.
The reference full-vector IVF recall on this cohort is 99.4921875%.
Each held-out row has768 timing samples, but only256 distinct test queries.

## Allocation and representation error

Four mixed variants were implemented: the illustrative32@8/32@4/64@2 schedule;
a variance-based separable allocation; an exact measured-MSE allocation; and a
measured-MSE allocation restricted to <=4 contiguous segments with boundaries
at multiples of eight PCA coordinates. All optimizers solve only their stated
additive objective, not optimal ANN recall, certificate traffic or CPU latency.

| Full-PCA128 scalar encoding | Learning reconstruction MSE | Change vs uniform4 | Mean base reconstruction norm | p95 norm |
|---|---:|---:|---:|---:|
| Uniform4 | 3214.5566 | reference | 56.5828 | 62.1492 |
| Manual32@8/32@4/64@2 | 15176.5102 | 372.12% worse | 122.8871 | 136.1695 |
| Variance allocation | 3768.7165 | 17.24% worse | 61.2565 | 66.9930 |
| Per-coordinate measured MSE | 2995.2488 | 6.82% lower | 54.6455 | 59.4052 |
| Grouped measured MSE | 3050.5569 | 5.10% lower | 55.1364 | 59.7836 |

The grouped optimizer chooses PCA coordinates1-8 at6 bits,9-120 at4 bits,
121-128 at2 bits. Thus8*6+112*4+8*2=512 bits, using only three segments.
The unrestricted coordinate optimizer gives6 bits to coordinates1,2,4,5,6;
2 bits to123-127; and4 bits to the remaining118 coordinates. This has six
contiguous decoder segments in the preserved PCA order. The variance allocation
is1@8,16@6,93@4,18@2 in PCA order. No eight-bit coordinates are selected by either
measured-MSE optimizer. Learning MSE, mean base norm and p95 norm are DIFFERENT
statistics; a6.82% squared-error reduction is not a6.82% radius reduction.

All mixed radii include errors from low-bit tail coordinates. The manual example
spends too many bits on its head for this quantizer; the large errors are an
observed result, not a failure of the covering invariant. Variance-only allocation
has worse reconstruction MSE, but can still improve shortlist recall at fixed R:
reconstruction distortion and neighbor-ranking quality are not interchangeable.
The high-resolution variance model does not exactly describe these deployed
min/max bins, especially at two bits. This test is not an evaluation of all
possible variable-precision schemes or an implementation of SAQ.

Original128/SQ4 has mean reconstruction norm32.8185 and plain PQ23.0165, both
well below the learned full-PCA mixed scalar norms. PCA64/SQ8 has represented-
space error3.0860, but omits64 directions; that number must not be directly
ranked against full-dimensional reconstruction errors.

## Fixed-R64 approximate held-out comparison

All candidates are ranked in RAM; only shortlisted physical pages are fetched.
All valid vectors on a fetched page are evaluated. No gap bridging is used.
These rows share shortlist size and fixed pages. All scalar rows use the SAME
new grouped scanner, including uniform controls. PQ uses the prior PQ LUT path.
Exact-match queries means identical ordered IDs to the full-vector IVF oracle.

| Representation | Global Recall@10 | Exact-match queries | Pages/query | Mean direct ms |
|---|---:|---:|---:|---:|
| PCA64/SQ8 | 99.2578125% | 250/256 | 55.4375 | 6.0819 |
| PCA128/SQ4 | 99.1796875% | 248/256 | 56.0039 | 11.6920 |
| Original128/SQ4 | 99.4921875% | 256/256 | 55.6172 | 11.6767 |
| Plain PQ64x8 | 99.4921875% | 256/256 | 55.5039 | 5.3031 |
| Mixed manual | 98.8281250% | 239/256 | 55.8867 | 11.3889 |
| Mixed variance | 99.4531250% | 255/256 | 55.7578 | 12.0437 |
| Mixed measured MSE | 99.3750000% | 252/256 | 55.8164 | 13.0919 |
| Mixed grouped MSE | 99.4140625% | 254/256 | 55.8242 | 11.9082 |

Measured-MSE allocation improves full-PCA scalar shortlist quality at the same
code size, but does not beat original-coordinate SQ or PQ here. It does not
reduce approximate page reads appreciably at fixed R. The relatively poor
manual R64 result is retained rather than hidden by a larger selected shortlist.
Any approximate arm matching all256 queries remains approximate, not certified.

## Development-selected approximate operating points

| Representation | Frozen R | Global Recall@10 | Exact-match queries | Pages/query | Mean direct ms |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 64 | 99.2578125% | 250/256 | 55.4375 | 6.0819 |
| PCA128/SQ4 | 64 | 99.1796875% | 248/256 | 56.0039 | 11.6920 |
| Original128/SQ4 | 32 | 99.3750000% | 253/256 | 28.7422 | 11.3741 |
| Plain PQ64x8 | 32 | 99.4921875% | 256/256 | 28.6719 | 4.9387 |
| Mixed manual | 128 | 99.4921875% | 256/256 | 106.7188 | 12.0542 |
| Mixed variance | 32 | 98.9843750% | 243/256 | 28.7617 | 11.6743 |
| Mixed measured MSE | 64 | 99.3750000% | 252/256 | 55.8164 | 13.0919 |
| Mixed grouped MSE | 64 | 99.4140625% | 254/256 | 55.8242 | 11.9082 |

The variance-based R32 setting misses the99% held-out target despite meeting it
on development. R64 is shown only as a predeclared fixed-size control, NOT a
retuned replacement. Manual coding needs R128 under the development selection,
reading roughly107 pages. Plain PQ reaches the reference answer on all test
queries at R32 and approximately29 pages; no mixed scalar configuration improves
its observed combination of latency, recall and verification traffic.

## Certified two-wave completion

Every row matches the full-vector selected-IVF-list oracle on all256 test queries,
with global Recall@10=99.4921875%. This is not globally exact ANN when the IVF
candidate lists omit points. All methods share the same exact SIMD FP64 scanner,
original FP32 payload and pooled rolling io_uring/O_DIRECT reader at depth16.

| Representation | Frozen R | Pages/query | Mean waves | Mean direct ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 128 | 182.7539 | 1.6094 | 9.8083 | 13.8294 |
| PCA128/SQ4 | 32 | 661.0859 | 1.9961 | 21.4238 | 32.4847 |
| Original128/SQ4 | 32 | 144.7891 | 1.9688 | 15.7462 | 19.9863 |
| Plain PQ64x8 | 32 | 71.3086 | 1.9531 | 8.3835 | 10.1475 |
| Mixed manual | 32 | 3083.8008 | 2.0000 | 40.7552 | 67.3481 |
| Mixed variance | 16 | 740.7305 | 2.0000 | 22.4727 | 34.6736 |
| Mixed measured MSE | 32 | 573.5742 | 1.9961 | 21.8610 | 31.8491 |
| Mixed grouped MSE | 32 | 574.3125 | 1.9961 | 20.6277 | 30.2174 |

The two learned-MSE variants have the SAME R32 as uniform PCA128, providing a
cleaner representation comparison. Grouped allocation reduces requested pages
13.13% and mean latency3.72%. Its three round ratios of mean latency are1.027,
1.042 and1.047. Unrestricted coordinate allocation reduces pages13.24%, but mean
latency is2.04% worse; all three rounds favor uniform4 over this more fragmented
encoding. Fewer bytes read is not automatically lower wall time.

Grouped MSE initially reads28.7852 pages and adds545.5273 completion pages.
Plain PQ initially reads28.6719 and adds42.6367. Page extents total385.5156 for
grouped MSE versus64.7656 for PQ. The reported wave count is a data-dependency
count, not an SSD command count; large waves can be split into bounded calls.
These results do not establish that mixed-bit coding is generally ineffective;
they establish a modest full-PCA scalar gain and a much stronger plain-PQ control
under this particular deployed quantizer and budget.

## Decoder control and CPU cost

The new scalar scanner handles fixed global bit segments natively, constructs
query tables inside timing and accumulates coordinates in FP64 order. It keeps
the previous shortlist/certificate semantics, including an upper-bound heap in
full-dimensional certified mode even though two-wave verification does not need
it. No full-index decoded-center cache is added. PQ needs64 group contributions;
full-dimensional scalar methods need128. These are not production FastScan
kernels and should not be described as optimal scalar/PQ implementation costs.

The generalized scanner itself is a REGRESSION on uniform PCA64 compared with
its old specialized LUT path, despite identical codes, answers and read plans:

| PCA64 implementation | Approx R64 direct ms | Certified R128 direct ms |
|---|---:|---:|
| Previous specialized LUT scanner | 5.0264 | 8.8214 |
| New grouped scalar scanner | 6.0819 | 9.8083 |

Do not credit PQ with beating an intentionally weakened baseline: the prior
PCA64 execution remains available, and the two old arms were rerun in this same
campaign. New scalar controls isolate allocation within a common grouped kernel;
they do not replace the old default. The approximately4.94ms PQ approximate mean
is close to the old PCA64's5.03ms, but PQ has higher observed recall and half its
verification pages in this cohort. The main signal is the resource/quality
tradeoff, not a fresh SOTA or large runtime win.

Direct rank/bound-construction averages are11.5525ms for uniform PCA128/R32,
12.8895ms for unrestricted mixed-MSE/R32 and11.7418ms for grouped-MSE/R32.
Corresponding read stages are5.2936,4.8410 and4.8308ms. Grouping retains most of
the distortion gain without as much decoding cost. It does not eliminate the
broad compressed scan. In memory replay, certified means are14.4150,14.2594 and
14.1938ms respectively; the direct/replay phases have different cache/pacing
conditions and are not subtraction-based estimates of pure device service time.

## Memory, validation and limits

Directory array accounts, including descriptors, are74.5138MiB for new PCA64,
74.5773MiB for uniform full-dimensional scalar,74.5773-74.5774MiB for mixed scalar,
and74.7022MiB for PQ. Shared descriptors are tiny and global: the grouped mixed
allocation uses128 width bytes plus60 segment bytes. There is no per-vector
precision metadata. Runtime, LUT/query scratch and read buffers remain extra.
The campaign read pool peaked at48078848bytes (45.852MiB) under its64MiB cap and
reported825245 allocation events across interleaved variants. That is not a
per-arm steady-state memory or allocation rate; it includes the costly manual
variant and reuse/eviction across the campaign. Peak experiment RSS was about
3.078GiB and includes base data, oracle, all variants and temporary build arrays.

All327 runner tests passed, with zero errors, failures or skips. The22 new tests
cover exact discrete optima, variable-width byte boundaries, fixed-size packing,
full decoded coverage, native ranking, uniform-code equivalence and certified
verification. Local focused tests passed22; an additional UBSan run passed80
randomized native rank/lower/upper-bound cases. A complete local regression run
did not finish within the available execution timeout; only the runner provides
the complete-suite result.

All eight representations audit every indexed vector after actual decoding.
There are164 initial mechanics comparisons and663275 post-search audited
rejections, using the first two development queries per applicable arm. These
reuse points and queries; they are not independent samples or a formal numerical
proof. All65280 timed searches preserve required exact-mode answers and repeated
method/query answers/counts. Approximate misses are measured, not asserted away.

Independent verification reconciled archive SHA256, all six source files,
65280 CSV rows,21760 saved neighbor records,126 aggregate rows, bit-budget and
byte/page equations, query disjointness, frozen development choices, reference
and global recall, uniform-code plan equivalence records and327 test cases.
The grouped MSE optimum was independently checked by exhaustive enumeration of
3686 feasible <=4-segment plans; a separate min-plus DP checks the coordinate
optimum. This is artifact analysis, not a second local full-SIFT benchmark.

The shared AMD Ryzen9 9955HX3D host has32 logical CPUs. CPU0 affinity does not
reserve a core. Load averages start21.091/20.403/20.343 and end21.827/21.708/21.162.
A file lock serializes GeoIVF speed campaigns, not other work. Brief read-only
progress helpers also ran. Frequency and CPU/device-cache state are uncontrolled.
Direct reads are genuine O_DIRECT, but the payload fits host RAM and base/oracle
arrays remain resident in the research process. Compare within this campaign,
not against older timings on different query cohorts or host loads.

No external DiskANN/Faiss/CLIP/SAQ benchmark, total-RSS cap, larger-than-RAM test,
query-concurrency result or independently instrumented physical NVMe-command
count is included. Bit plans are optimal only under the tested objective and
2/4/6/8-bit/global-minmax constraints. Learned bins, per-list calibration,
odd widths and rearranging same-width dimensions for cheaper decoding remain
unmeasured. No novelty claim is made for unequal bit allocation itself.

## Decision

Retain the mixed-bit encoder and allocation tools as opt-in experimental paths.
Measured-error allocation modestly improves full-PCA scalar reconstruction and
certified page traffic, whereas variance-only and aggressive head/tail allocation
are unreliable guides under the current quantizer. The grouped plan gives the
cleanest scalar tradeoff. Plain full-dimensional PQ remains the stronger observed
reference; keep the old PCA64 decoder rather than replacing it with the slower
generalized scanner. Reproduction: docs/MIXED_BITS.md and scripts/qualify_mixedbits.py.
Preserve the downloaded artifact beyond GitHub's30-day retention period.
