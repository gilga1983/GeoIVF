# Equal-budget rotation and full-space verification results, 2026-09-30

## Provenance and experimental scope

Tested implementation: `b41724448f49f5d25c4ed0fb51601c15d1161915`.
Successful workflow: `36655355595`; job `109698417919`.
Runner: geoivf-gilga-Legion-Pro-7-16AFR10H-01.
Artifact: `11072257558`, rotations-36655355595.
Downloaded archive SHA256:
`90665f26a865a38903fd05893be7ee16f319ebf2ca5ae53c7411297516d4fa54`.

Full canonical SIFT1M: 1,000,000 unchanged 128D FP32 vectors, eight vectors per
4 KiB page, 125445 pages, nlist=1024, nprobe=64, k=10, construction seed12345.
Payload SHA256 remains
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
All representations use the SAME page contents/order and candidate lists.
PCA is fitted on the separate 100000-vector learning file. PQ/OPQ use a fixed
50000-vector learning subset, never test queries. Scalar ranges use base-corpus
min/max, so training/calibration sources are explicitly different for SQ and PQ.

Development: 128 queries, permutation(seed=20260929)[:128]. All 78 development
arms run three randomly interleaved direct-I/O rounds. Shortlist sizes are
16/32/64/128/256. Choose the fastest >=99% strict Recall@10 setting separately
for each representation and approximate/certified mode. Choices are frozen
before evaluating NEW held-out queries permutation[1512:1768]. These 256 queries
exclude the entire previous development pool and both earlier test cohorts.

Held-out: 25 predeclared/frozen arms, three rounds each of real direct I/O and
RAM replay. These include fixed-R64 approximate comparisons for every encoding,
all full-dimensional one-wave upper-bound variants, and three legacy controls.
All exact reference answers have strict global Recall@10=99.375% on this cohort.
Each held-out table row has 768 timing samples but only 256 distinct queries.
No test-set parameter adjustment occurs. The fixed nprobe is NOT jointly tuned
with the code family or shortlist in this representation experiment.

## Encoding and execution changes

Each new representation uses 64 code bytes per vector plus an outward-rounded
four-byte actual reconstruction-error radius: 544 geometry bytes/full page.
The common eight-byte radial interval and shared/structural metadata are extra.
No inverse rotation is used during search. Original vectors are fetched and
verified in their original space. Means, matrices, scalar quantizers and PQ
codebooks remain shared, not expanded into per-vector floating-point copies.

Seven code families: PCA64/SQ8; the same PCA64 subspace followed by a seeded
random rotation/SQ8; centered original128/SQ4; PCA128/SQ4; random128/SQ4; PQ64x8;
and OPQ followed by PQ64x8. PQ has 64 two-dimensional, 256-centroid codebooks.

PQ and OPQ training/encoding use Faiss 1.15.1. The OPQ pilot has 20 outer
iterations, 20 initial PQ iterations and four subsequent PQ iterations, then a
separate final 20-iteration PQ fit. Plain PQ has the same final training subset,
seed and iteration budget. This is NOT exhaustive OPQ tuning or its default
training schedule. OPQ must not be declared universally inferior from this run.

All NEW representations use a common native FP64 lookup-table ranker, including
query transformation and table construction INSIDE measured query time. This is
not Faiss FastScan or an optimized product-quantized index benchmark. An SQ128
scan has 128 scalar contributions and four-bit unpacking; a PQ64 scan has 64
lookup contributions. The old PCA64 direct-arithmetic ranker is rerun separately
to isolate the execution change. New and old PCA64 packed code bytes match
exactly; numeric guards/shared matrix storage differ slightly.

Verification uses the SAME SIMD FP64 exact scanner and pooled rolling
io_uring/O_DIRECT reader at queue depth16. New RAM-first arms do not bridge gaps.
The legacy staged control retains window256/gap2. Bounds certify all projected
points after decoding actual stored records, not ideal unquantized centers.

## 1. Rotation alone and fixed-shortlist ranking

All rows below use R=64, one approximate verification wave, and the common new
ranker. These fixed-R64 comparisons were specified before test evaluation.
Exact-match queries means all ten returned IDs match exhaustive full-vector
search over the selected IVF lists; it is stronger than global aggregate recall.

| Representation | Strict global Recall@10 | IVF-reference Recall@10 | Exact-match queries | Pages/query | Direct mean ms |
|---|---:|---:|---:|---:|---:|
| PCA64, SQ8 | 99.2578125% | 99.8828125% | 254/256 | 56.1992 | 4.7050 |
| PCA64 + random rotation, SQ8 | 99.2578125% | 99.8828125% | 254/256 | 56.1289 | 4.7081 |
| Original128, SQ4 | 99.3750000% | 100.0000000% | 256/256 | 56.3594 | 10.0408 |
| Full PCA128, SQ4 | 98.8281250% | 99.4531250% | 244/256 | 56.8672 | 10.0157 |
| Random128, SQ4 | 99.2968750% | 99.9218750% | 254/256 | 56.5703 | 9.9793 |
| Plain PQ64x8 | 99.3750000% | 100.0000000% | 256/256 | 56.2617 | 4.9643 |
| OPQ + PQ64x8 | 99.3750000% | 100.0000000% | 256/256 | 56.2500 | 4.9937 |

A random rotation within the existing PCA64 subspace gives no useful gain here.
At this code budget, retaining all original coordinates can improve shortlist
quality, but the uniform rotated scalar encodings are not automatically better.
PCA128/SQ4 misses its predeclared 99% held-out target; it had met that target on
development. It was NOT retuned after this failure.

The full-dimensional PQ codes recover all candidate-set answers at R64 in this
sample, without the roughly doubled scan work of the current SQ128 ranker.
Neither approximate configuration has a universal no-additional-loss guarantee.

## 2. Development-selected approximate operating points

| Representation | Frozen R | Mean ms | Strict Recall@10 | IVF-reference matches | Pages/query |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 64 | 4.7050 | 99.2578125% | 254/256 | 56.1992 |
| PCA64 + random/SQ8 | 64 | 4.7081 | 99.2578125% | 254/256 | 56.1289 |
| Original128/SQ4 | 32 | 9.6452 | 99.3359375% | 255/256 | 29.0703 |
| PCA128/SQ4 | 64 | 10.0157 | 98.8281250% | 244/256 | 56.8672 |
| Random128/SQ4 | 64 | 9.9793 | 99.2968750% | 254/256 | 56.5703 |
| Plain PQ64x8 | 32 | 4.6469 | 99.3750000% | 255/256 | 28.9805 |
| OPQ + PQ64x8 | 32 | 4.7017 | 99.3750000% | 256/256 | 28.9102 |

Plain PQ and OPQ select half the shortlist and read about half as many pages as
PCA64 while maintaining the full-IVF global recall on these test queries. For
plain PQ/R32, one reference neighbor differs on query2311; that missed ID is not
in the supplied global top10, so aggregate global recall remains unchanged.
OPQ/R32 happens to match all reference answers in this sample. Do not describe
approximate PQ/R32 as certified or exactly equivalent just because recall matches.

The direct mean times of PCA64, PQ and OPQ selected approximate settings are
close (4.65-4.71 ms). The 1.24% nominal mean difference between plain PQ/R32 and
PCA64/R64 is not a persuasive runtime advantage on this shared host. The useful
signal is the ranking/verification-traffic tradeoff, not a new speed headline.

The previous PCA64 arithmetic ranker takes5.2812 ms at R64 on these SAME queries,
versus4.7050 ms for the new common table ranker, with identical answers/pages.
That 10.91% mean improvement is an implementation change, not an effect of a
new rotation. RAM replay likewise changes3.5505 to2.8302 ms. Do not attribute it
to quantization geometry.

## 3. Certified completion: plain full-dimensional PQ is promising

All rows below return the independent IVF-candidate-set answer on ALL256 test
queries, with global Recall@10=99.375%. Shortlist sizes were selected on development.

| Representation | Frozen R | Pages/query | Mean decision waves | Direct mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 128 | 201.1133 | 1.6836 | 8.6465 | 12.0961 |
| PCA64 + random/SQ8 | 128 | 202.7813 | 1.6914 | 8.6573 | 12.1499 |
| Original128/SQ4 | 32 | 163.6719 | 2.0000 | 13.9690 | 17.2322 |
| PCA128/SQ4 | 32 | 768.0781 | 2.0000 | 20.4605 | 32.5001 |
| Random128/SQ4 | 32 | 812.2344 | 2.0000 | 20.8960 | 34.1338 |
| Plain PQ64x8 | 16 | 78.5625 | 2.0000 | 8.1054 | 9.4771 |
| OPQ + PQ64x8 | 16 | 153.6563 | 2.0000 | 8.9312 | 11.0159 |

Plain PQ's selected certified point requests60.94% fewer pages than the selected
PCA64 point, with6.26% lower mean time in this run. These rows use different R;
they describe frozen operating points, NOT a same-R causal isolation. The
same-R32 DEVELOPMENT control also favors PQ:72.46875 versus176.75 pages/query,
both returning the same IVF answer. Its means are7.8212 versus8.4253 ms.

On held-out queries, PQ/R16 first fetches14.8711 pages and certification adds
63.6914; every query uses a second wave. It averages71.0352 extents/query.
The old arithmetic-certified R32 control requests195.0859 pages at9.1088 ms,
and the staged control requests428.3477 pages at13.1297 ms in this campaign.
These are fresh same-campaign controls, not copied from previous runtime reports.

The larger nominal PCA64 quantization accuracy does not make its bound tighter
in full space: its radius describes only retained-coordinate error. Full PQ has
larger reconstruction-error radii but no discarded dimensions. Ranking quality
and bound tightness are different objectives, as the OPQ result also illustrates.

## 4. One-wave full-space upper bounds are valid but too loose here

For deployed transform T, certify beta>=||T||2 and, for full square transforms,
0<alpha<=sigma_min(T). If r covers decoded error and d is query-to-code distance,
use max(0,(d-r-guard)/beta) as a lower bound and (d+r+guard)/alpha as an upper bound.
The kth smallest upper bound on DISTINCT candidate vectors is a valid pre-read
threshold. Fetch all pages whose lower bounds do not exceed it, then rank their
original vectors. This is one data-dependent wave, not one SSD command.

| Full-dimensional code | Pages/query | Direct mean ms | Decision waves | Mean reader calls |
|---|---:|---:|---:|---:|
| Original128/SQ4 | 1503.1172 | 25.9688 | 1 | 1.0664 |
| PCA128/SQ4 | 5479.1445 | 48.3721 | 1 | 2.2656 |
| Random128/SQ4 | 5929.8672 | 49.4467 | 1 | 2.4297 |
| Plain PQ64x8 | 406.5820 | 10.1299 | 1 | 1.0000 |
| OPQ + PQ64x8 | 1035.2031 | 16.2194 | 1 | 1.0000 |

All one-wave certificates preserve every tested IVF answer. But even plain PQ
requires5.18 times as many pages as its selected two-wave verifier. The initial
read is valuable because its exact threshold is much tighter. Eliminating one
more decision wave is not worth that traffic inflation in this setup. Large
one-wave plans are safely split into bounded batches without recomputing their
admission threshold; this is why reader calls can exceed decision waves.

## 5. Distortion, codebook training and memory

| Encoding | Mean decoded error in represented space | p95 error | Omitted-space norm mean | Build seconds | Directory arrays MiB |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 3.0860 | 3.4335 | 96.9419 | 3.49 | 74.5137 |
| PCA64 + random/SQ8 | 3.2256 | 3.5247 | 96.9419 | 3.44 | 74.5137 |
| Original128/SQ4 | 32.8185 | 37.3961 | 0 | 6.05 | 74.5772 |
| PCA128/SQ4 | 56.5828 | 62.1492 | 0 | 5.66 | 74.5772 |
| Random128/SQ4 | 59.2215 | 63.1040 | 0 | 5.51 | 74.5772 |
| Plain PQ64x8 | 23.0165 | 29.9389 | 0 | 56.40 | 74.7022 |
| OPQ + PQ64x8 | 31.9869 | 40.9954 | 0 | 288.03 | 74.7022 |

PCA64 error norms exclude omitted coordinates and are NOT directly comparable
with full-dimensional reconstruction errors. All models have the same code plus
radius budget; full PQ adds about0.19 MiB shared arrays versus the rectangular
PCA64 model. Total runtime and read/query scratch are additional. PQ codebooks
are not free. New PCA64 stores a128x64 matrix rather than the previous complete
128x128 matrix, accounting for most of its small directory-size reduction.

For this base corpus and min/max uniform quantizer, full rotations increase
scalar reconstruction error relative to original coordinates. A possible reason
is their different coordinate ranges/tails; this is an interpretation rather
than an isolated causal experiment. The original axes are a necessary control.

The trained OPQ variant does NOT beat plain PQ on decoded error or certified
traffic under this fixed training budget, although it has excellent approximate
shortlist outcomes. This warrants convergence/initialization/training checks
before making any general statement about OPQ. Learned rotation is not a
universal guarantee for a finite training procedure. The plain-PQ advantage
shows that full-dimensional learned codebooks, not rotation alone, are useful
in this experiment.

Build times above include transformation-specific learning, projection, encoding,
packing, coverage checks and sidecar writing. They exclude shared IVF/layout and
initial PCA fitting. Thus they are not complete normalized system-build costs.

## 6. Runtime decomposition and experimental limits

Direct approximate PCA64/R64 spends3.2708 ms in ranking, versus3.5134 ms for
PQ/R32. Read-stage time is0.9246 versus0.7462 ms. Better candidate quality saves
I/O, but the ranker remains most of the query. The current full-dimensional SQ4
ranker takes roughly8.5 ms and should not be treated as optimized scalar-quantizer
performance. All table construction and unpacking are charged.

Certified PQ/R16 spends4.9218 ms ranking/bound computation,1.6522 ms in reads,
and additional certificate/coalescing/scanning work. Its RAM-replay mean is
4.9904 ms versus4.9055 for PCA64/R128. Fewer verification pages do not eliminate
the CPU cost of scoring the broad IVF candidate set. The full-dimensional
certified ranker currently also computes the k-smallest upper-bound heap even
for two-wave mode; removing unnecessary work is not evaluated here.

Host is the shared AMD Ryzen9 9955HX3D laptop, 32 logical CPUs; one process is
pinned to CPU0, not an exclusive core. Timing-start load averages18.856/18.541/
17.461, end16.949/17.389/17.450. CPU frequency, other workloads and device cache
are uncontrolled. Only GeoIVF speed campaigns are serialized by the host lock.
The light read-only progress jobs inspect the benchmark's logs, not its inputs.

The roughly490 MiB payload fits host RAM, while base/oracle/multiple variants
coexist in this research process. Actual direct reads use O_DIRECT, but this
is not cold-device or larger-than-RAM evidence. Peak benchmark RSS is about
3.01 GiB and is NOT deployment memory. Common read pool peak is6475776 bytes
(6.176 MiB); it is not a per-arm steady-state measurement. No new external
Faiss/CLIP/DiskANN run, concurrent QPS, matched whole-process memory cap or
physical NVMe-command instrumentation is included.

## 7. Validation and artifact reconciliation

All305 runner tests passed, no errors, failures or skips. There are22 new
rotation/PQ tests beyond the preceding283-test suite, including full-space
upper/lower bound safety, rectangular-transform handling, scalar/PQ code budgets,
actual decoded coverage, and certified answers. Local suite:284passed/21 optional
dependency skips. An additional local UBSan check passed100 randomized exact
verification cases with no errors. Local Faiss tests were not run; the runner
executed them, including PQ and OPQ.

All seven built encodings audited coverage of every indexed vector. The first
two development queries per arm supply156 mechanics checks, including695900
post-search audited rejection decisions. These repeat vectors/pages across
encodings and are not independent samples or formal floating-point proofs.

68352 timed searches were recorded:29952 development-direct,19200 heldout-direct,
and19200 heldout-memory. All exact-mode queries return the independent FP64
candidate-set oracle answer, and all repeated method/query IDs and operation
counts agree. Approximate discrepancies are retained and measured, not asserted
away. There are384 distinct queries,128development plus256test.

After download, archive SHA matched GitHub. All six new executable/workflow
files matched the locally tested source hashes. An independent artifact script
reconciled68352 CSV rows,22784 stored neighbor records and128 aggregate rows:
query exclusions, recall/IVF recall, exact-match flags, all page/byte/wave equations,
repeated counts, round means, medians/p95, and development-only frozen selection.
It also checked305 test cases and all model code-budget/coverage metadata.
These checks analyze archived evidence, not a second local SIFT1M run.

## Decision

Do not make rotation itself the leading optimization. Retain PCA64 as the
established low-overhead reference and carry plain full-dimensional PQ with our
page verifier as a promising equal-code-budget alternative. It can halve the
approximate shortlist and substantially reduce certified traffic; its runtime
advantage is modest because RAM scoring still dominates. Keep OPQ and the full
uniform scalar encodings as measured alternatives, not a universal ranking.

The full-space upper-bound construction is correct and reproducible, but a
small first exact verification wave gives a better traffic/latency tradeoff
than the tested all-RAM one-wave certificate. These findings support separating
representation quality, verification contract and scoring implementation.
Reproduction: docs/ROTATION_BUDGET.md and scripts/qualify_rotations.py. Preserve
the downloaded artifact beyond GitHub's30-day retention period.
