# Equal-budget rotation and full-space verification results, 2026-09-30

## Provenance

Tested implementation: `b41724448f49f5d25c4ed0fb51601c15d1161915`.
Successful workflow: `36655355595`; job: `109698417919`.
Runner: `geoivf-gilga-Legion-Pro-7-16AFR10H-01`.
Artifact: `11072257558`, `rotations-36655355595`.
Downloaded ZIP SHA256:
`90665f26a865a38903fd05893be7ee16f319ebf2ca5ae53c7411297516d4fa54`.

This is a completed canonical SIFT1M experiment, not a prediction. It uses all
1,000,000 unchanged 128D FP32 vectors, eight vectors per 4 KiB page, 125445 pages,
nlist=1024, nprobe=64, k=10 and construction seed12345. All methods use the same
GeoPack page membership/order and IVF candidate lists. The payload SHA256 is
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.

Development uses 128 queries from permutation(seed=20260929)[:128]. The 78 arms
run three randomly interleaved direct-I/O rounds. R is swept over16/32/64/128/256.
The fastest setting meeting 99% strict Recall@10 is frozen separately for every
representation and approximate/certified mode. This does not jointly tune nprobe.

The fresh held-out cohort is permutation[1512:1768], disjoint from the entire
previous development pool and both previous 256-query test cohorts. Its25 arms
include frozen choices, predeclared approximate R64 controls, all applicable
one-wave upper-bound variants and three legacy controls. Held-out direct-I/O and
RAM-replay phases each have three randomized interleaved rounds. Each reported
held-out row has768 timing samples but only256 distinct queries. No test retuning.
Full-vector IVF's global strict Recall@10 is99.375% on this new cohort.

## Representations and fairness

Every representation stores64 code bytes/vector plus one outward-rounded FP32
reconstruction-error radius:544 geometry bytes/full page. The common radial
interval, shared matrices/codebooks and structural metadata are extra. Original
SSD vectors are unchanged; queries are transformed, not candidates inverse-rotated.

The seven families are PCA64/SQ8; the same PCA64 subspace with a seeded random
rotation/SQ8; original128/SQ4; PCA128/SQ4; random128/SQ4; plain PQ64x8; OPQ+PQ64x8.
PQ has64 two-dimensional codebooks, each containing256 codewords.

PCA uses the separate100000-vector learning file. PQ/OPQ use the same fixed50000
learning vectors. Scalar min/max ranges are calibrated on the indexed base data.
No query trains the representation. OPQ has20 outer iterations,20 initial PQ
iterations and four subsequent PQ iterations, then a separately trained final
20-iteration PQ. Plain PQ uses the same final training budget and seed. This is
not exhaustive OPQ tuning or its default schedule, so no universal OPQ ranking
is justified by this run.

Faiss1.15.1 trains and encodes PQ/OPQ. All NEW representations use our common
native FP64 lookup-table scanner, not Faiss FastScan. Table construction and
query transformation are inside timing. SQ128 needs128 coordinate contributions
and four-bit unpacking; PQ64 needs64 table contributions. Shared codebook costs
are recorded separately. No expanded per-vector reconstruction table is retained.

Old arithmetic-ranked PCA64 approximate/certified modes and the prepared staged
control are rerun. New and old PCA64 packed codes match exactly; shared matrix
shape and numerical guards differ slightly. This prevents a new LUT kernel's
benefit from being attributed to rotation. All arms use the same SIMD FP64 exact
scanner and pooled rolling io_uring/O_DIRECT reader, queue depth16. RAM-first
verification uses no gap bridging; legacy staged uses window256/gap2.

## Fixed-R64 held-out approximate comparison

These fixed-size comparisons were specified before test evaluation. Exact-match
queries means all ten IDs equal the exhaustive answer over the selected IVF
lists, which is stronger than matching aggregate global recall.

| Representation | Global Recall@10 | IVF-reference Recall@10 | Exact-match queries | Pages/query | Direct mean ms |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 99.2578125% | 99.8828125% | 254/256 | 56.1992 | 4.7050 |
| PCA64 + random/SQ8 | 99.2578125% | 99.8828125% | 254/256 | 56.1289 | 4.7081 |
| Original128/SQ4 | 99.3750000% | 100.0000000% | 256/256 | 56.3594 | 10.0408 |
| PCA128/SQ4 | 98.8281250% | 99.4531250% | 244/256 | 56.8672 | 10.0157 |
| Random128/SQ4 | 99.2968750% | 99.9218750% | 254/256 | 56.5703 | 9.9793 |
| Plain PQ64x8 | 99.3750000% | 100.0000000% | 256/256 | 56.2617 | 4.9643 |
| OPQ + PQ64x8 | 99.3750000% | 100.0000000% | 256/256 | 56.2500 | 4.9937 |

Random rotation within PCA64 adds no useful gain here. Full-dimensional coding
can improve shortlist quality, but arbitrary rotation and coarse scalar coding
do not automatically improve it. PCA128/SQ4 fails the predeclared99% held-out
target despite meeting it on development. That failure is retained, not retuned.

PQ/OPQ recover every IVF-reference answer at R64 on this sample, without the
roughly doubled scoring work of our current SQ128 implementation. Approximate
mode still has no universal guarantee merely because this sample passed.

## Development-selected approximate operating points

| Representation | Frozen R | Direct mean ms | Global Recall@10 | Exact-match queries | Pages/query |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 64 | 4.7050 | 99.2578125% | 254/256 | 56.1992 |
| PCA64 + random/SQ8 | 64 | 4.7081 | 99.2578125% | 254/256 | 56.1289 |
| Original128/SQ4 | 32 | 9.6452 | 99.3359375% | 255/256 | 29.0703 |
| PCA128/SQ4 | 64 | 10.0157 | 98.8281250% | 244/256 | 56.8672 |
| Random128/SQ4 | 64 | 9.9793 | 99.2968750% | 254/256 | 56.5703 |
| Plain PQ64x8 | 32 | 4.6469 | 99.3750000% | 255/256 | 28.9805 |
| OPQ + PQ64x8 | 32 | 4.7017 | 99.3750000% | 256/256 | 28.9102 |

PQ/OPQ select half the shortlist and read about half as many pages as PCA64 at
similar mean latency. Plain PQ/R32 misses one IVF-reference ID on query2311, but
that ID is not in the supplied global top10; global recall remains unchanged.
OPQ/R32 happens to recover every reference answer. Neither approximate arm is
certified. The nominal1.24% speed difference between selected plain PQ and PCA64
is not a convincing runtime win on this shared host; traffic and ranking quality
are the useful signals.

The old arithmetic PCA64/R64 ranker takes5.2812 ms on these same queries versus
4.7050 ms with the new table scanner, with identical answers/page counts. Its
10.91% mean improvement is an implementation effect, not a new geometry. RAM
replay likewise changes3.5505 to2.8302 ms.

## Certified two-wave completion

All rows return exactly the full-vector IVF-candidate-set answer on all256 held-out
queries, at global Recall@10=99.375%. Shortlist choices are development-only.

| Representation | Frozen R | Pages/query | Decision waves | Direct mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 128 | 201.1133 | 1.6836 | 8.6465 | 12.0961 |
| PCA64 + random/SQ8 | 128 | 202.7813 | 1.6914 | 8.6573 | 12.1499 |
| Original128/SQ4 | 32 | 163.6719 | 2.0000 | 13.9690 | 17.2322 |
| PCA128/SQ4 | 32 | 768.0781 | 2.0000 | 20.4605 | 32.5001 |
| Random128/SQ4 | 32 | 812.2344 | 2.0000 | 20.8960 | 34.1338 |
| Plain PQ64x8 | 16 | 78.5625 | 2.0000 | 8.1054 | 9.4771 |
| OPQ + PQ64x8 | 16 | 153.6563 | 2.0000 | 8.9312 | 11.0159 |

The selected plain-PQ point requests60.94% fewer pages than selected PCA64, with
6.26% lower mean latency. These are frozen operating points with different R,
not a same-R causal isolation. A same-R32 DEVELOPMENT control also favors PQ:
72.46875 versus176.75 pages, at7.8212 versus8.4253 ms, with identical IVF answers.

Held-out PQ/R16 initially reads14.8711 pages and completion adds63.6914. Every
query requires the second wave; average extent count is71.0352. The prior
arithmetic-certified R32 control takes9.1088 ms/195.0859 pages, while staged
search takes13.1297 ms/428.3477 pages in this same campaign.

PCA64's radius measures error only in retained coordinates. Full-dimensional PQ
retains omitted directional information and supports a full-space error bound.
Its larger quantization radius therefore does not imply a weaker certificate.
OPQ's good shortlist recall but poorer certification illustrates that ranking
quality and conservative-bound tightness are different optimization objectives.

## One-wave upper-bound certificate

For transform T, beta bounds its largest singular value above, while alpha bounds
its smallest singular value below only for full square transforms. A measured
reconstruction-error radius r and query-to-code distance d give guarded bounds
max(0,(d-r-guard)/beta) and (d+r+guard)/alpha. For rectangular PCA64, alpha=0;
its projected upper bound must NOT be used as an original-space upper bound.

The kth smallest valid upper bound among distinct candidates supplies a pre-read
threshold. Fetch every page whose lower bound does not exceed it and rank original
vectors. All tested answers pass, but the threshold is often very loose.

| Full-dimensional code | Pages/query | Direct mean ms | Decision waves | Mean reader calls |
|---|---:|---:|---:|---:|
| Original128/SQ4 | 1503.1172 | 25.9688 | 1 | 1.0742 |
| PCA128/SQ4 | 5479.1445 | 48.3721 | 1 | 2.2695 |
| Random128/SQ4 | 5929.8672 | 49.4467 | 1 | 2.4258 |
| Plain PQ64x8 | 406.5820 | 10.1299 | 1 | 1.0000 |
| OPQ + PQ64x8 | 1035.2031 | 16.2194 | 1 | 1.0000 |

Even plain PQ reads5.18 times as many pages as its selected two-wave verifier.
The first exact batch is valuable because it supplies a much tighter threshold.
One decision wave is not one SSD command. Large plans are split into bounded
reader calls without updating the admission threshold, preserving that contract.

## Distortion, build cost, and resident arrays

| Encoding | Mean represented-space error | p95 error | Mean omitted norm | Build s | Directory MiB |
|---|---:|---:|---:|---:|---:|
| PCA64/SQ8 | 3.0860 | 3.4335 | 96.9419 | 3.49 | 74.5137 |
| PCA64 + random/SQ8 | 3.2256 | 3.5247 | 96.9419 | 3.44 | 74.5137 |
| Original128/SQ4 | 32.8185 | 37.3961 | 0 | 6.05 | 74.5772 |
| PCA128/SQ4 | 56.5828 | 62.1492 | 0 | 5.66 | 74.5772 |
| Random128/SQ4 | 59.2215 | 63.1040 | 0 | 5.51 | 74.5772 |
| Plain PQ64x8 | 23.0165 | 29.9389 | 0 | 56.40 | 74.7022 |
| OPQ + PQ64x8 | 31.9869 | 40.9954 | 0 | 288.03 | 74.7022 |

PCA64 errors exclude omitted coordinates, so they are not comparable directly to
full-dimensional errors. Shared matrices/codebooks are charged: PQ adds about
0.19 MiB versus the rectangular PCA64 directory, not zero bytes. New PCA64 stores
a128x64 matrix rather than the old128x128 one. Runtime and query/read scratch are
extra. All approximate arms still retain the common radius array.

Rotated full-dimensional uniform quantizers have larger error than the original
axes here. Different axis ranges/tails are a possible explanation, not an isolated
causal result. The OPQ configuration also has larger mean error and certification
traffic than plain PQ under this pilot budget. Further convergence/initialization
checks are needed before treating that as representative of optimized OPQ.

Build seconds include model-specific learning, projection, encoding, packing,
cover audits and writing. They exclude shared IVF/page construction and initial
PCA fitting, so they are not complete normalized system-build costs.

## CPU and limits

Approximate PCA64/R64 spends3.2708 ms ranking versus3.5134 ms for PQ/R32; read
stages cost0.9246 versus0.7462 ms. Better shortlists reduce storage traffic but
ranking still dominates. Our SQ128 ranker costs roughly8.5 ms and is not an
optimized production scalar-quantization baseline.

Certified PQ/R16 spends4.9218 ms ranking/bound construction and1.6522 ms in reads,
plus certificate/coalescing/exact scanning. RAM-replay mean is4.9904 ms versus
PCA64/R128's4.9055 ms. The full-dimensional certified ranker currently also
maintains the upper-bound heap in two-wave mode, where it is unnecessary work.
No speedup from removing that work is claimed in this first implementation.

Shared AMD Ryzen9 9955HX3D host,32 logical CPUs; CPU0 affinity does not reserve an
exclusive core. Load averages start18.856/18.541/17.461 and end16.949/17.389/17.450.
Frequency, unrelated workloads and device caching are uncontrolled. A file lock
serializes GeoIVF timing campaigns only. Brief progress jobs read experiment logs.
The payload fits host RAM; original data/oracle/all variants coexist in the
research process. Peak RSS is about3.01 GiB, NOT deployment memory. The common
read pool reserves6475776 bytes (6.176 MiB), not a per-arm steady-state figure.

Direct reads genuinely use O_DIRECT, but this is not a cold-device or larger-than-
RAM test. There is no new Faiss/CLIP/DiskANN timing run, matched RSS cap, concurrent
QPS measurement, or physical NVMe-command instrumentation. Compare timings within
this campaign, not against earlier runs on different cohorts and machine loads.

## Validation

All305 runner tests pass with no errors, failures or skips, including22 new
rotation/PQ tests. Local suite:284passed,21 optional dependency skips. Additional
local UBSan tests passed100 randomized exact-verification cases. Faiss/PQ/OPQ tests
were executed on the runner, not locally where Faiss was unavailable.

All seven encodings audit coverage of every indexed vector after decoding actual
packed records. The first two development queries per arm supply156 mechanics
checks and695900 post-search rejected-page audits. These repeat pages/encodings,
not independent vector samples or formal floating-point proofs.

68352 timed searches comprise29952 development-direct,19200 heldout-direct and
19200 heldout-memory executions. All exact-mode results match the independent
FP64 candidate-set oracle; repeated answers/counts agree. Approximate differences
are measured. There are384 distinct queries,128development plus256test.

Independent artifact verification reconciles ZIP SHA256, all six source files,
68352 CSV rows,22784 saved neighbor records,128 aggregate rows, disjoint query
sets, global/reference recall, exact-match flags, every page/byte/wave equation,
repeated counts, timing means/medians/p95/rounds, frozen development choices,
code-budget/coverage metadata and305 test cases. This is archived-evidence
analysis, not a second local canonical benchmark.

## Decision

Rotation itself is not the leading improvement here. Keep PCA64 as a reference
and carry plain full-dimensional PQ with the page verifier as a promising equal-
code-budget alternative. It can halve the approximate shortlist and substantially
reduce certified traffic; runtime changes remain modest because scoring dominates.
OPQ and rotated scalar codecs remain explicit ablations, not a universal ranking.

Full-space one-wave certification is valid but less efficient than obtaining a
tight threshold from a small initial exact batch in this experiment. Separating
representation, verification contract, and ranker implementation remains useful.
Reproduce via docs/ROTATION_BUDGET.md and scripts/qualify_rotations.py; preserve
the artifact beyond GitHub's30-day retention period.
