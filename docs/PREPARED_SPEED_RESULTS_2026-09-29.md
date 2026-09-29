# Prepared query planner speed results, 2026-09-29

## Provenance and scope

Tested implementation: `63022d9d78a693f866c815f4eaf98f6b661b2b4c`.
Successful workflow: `36570881942`; job: `109414287740`.
Runner: `geoivf-gilga-Legion-Pro-7-16AFR10H-04`.
Artifact: `11034855012`, `prepared-speed-36570881942`.
Downloaded ZIP SHA256:
`75b4f22440efc0f13a8336b2fdc38407b9393ee07c9d9af4dc8459b094c9ac38`.

Canonical SIFT1M, all 1M 128D FP32 vectors, 4 KiB pages, eight vectors/page,
125445 pages, nlist=1024, nprobe=64, k=10, seed=12345. Same frozen GeoPack
payload, learning-file PCA64, independent 6/8-bit codes and measured-error radii.
The 64 development queries are the same as the preceding correctness cohorts.
This campaign TIMES ALL 64, versus 32 in the preceding native-speed report.
All seven arms are rerun together in seven randomly interleaved rounds in each
of two phases: memory replay and real pooled io_uring/O_DIRECT at queue depth 16.
Use comparisons WITHIN this campaign, not raw timings across different cohorts.

Payload SHA256: `8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA fingerprint: `d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Candidate routes: `16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.
All match preceding canonical experiments. Recall@10 is 0.99375 for every arm.

## Changes

The new query-local prepared context eliminates repeated per-window query
validation, query-equality comparisons, guard calculation and per-list radial
norms. The preceding path already cached the projected query; this is NOT a
claim that it previously performed a full PCA rotation for every window.

A native function receives a contiguous page interval and performs radial
rejection, adaptive ball selection, and extent coalescing in one call. It uses
reusable mask/extent scratch and compile-time 4/6/8-bit decoders. It avoids the
intermediate NumPy page-ID/mask extraction pipeline and Python sorted-set
coalescing. No expanded full-index centers, lookup tables, new radius formats,
changed summaries, or hand-written SIMD are introduced.

The current online kth threshold still comes only from fetched FP32 vectors.
The same native FP64 scan, top-k heap, pooled reader, seeding, 64-page filtered
windows, zero-gap coalescing and maximum extent size remain in use. Unfiltered
baselines are allowed whole-list reads. An unfiltered prepared control receives
the same coalescing improvement, so the final comparison does not withhold it.
The opt-in CLI is `--selection prepared --scan native`; defaults are unchanged.

## Real asynchronous direct-I/O results

Wall milliseconds per complete online query, including fresh coarse routing,
projection/preparation, filtering/coalescing, reads, exact scans and top-k.
Each row has 448 timed samples but only 64 DISTINCT development queries.

| Configuration | Mean ms | Median ms | p95 ms | Pages/query |
|---|---:|---:|---:|---:|
| Unfiltered native, prior planner | 50.5833 | 50.3794 | 60.0658 | 8394.7969 |
| Unfiltered native, prepared planner | 48.8653 | 48.3863 | 58.8565 | 8394.7969 |
| PCA64 8-bit, full native bounds | 14.5648 | 13.9898 | 23.9875 | 297.9844 |
| PCA64 8-bit, prior adaptive selection | 14.5664 | 14.1862 | 24.4839 | 297.9844 |
| PCA64 8-bit, prepared selection | 9.3399 | 8.4685 | 18.9376 | 297.9844 |
| PCA64 6-bit, prior adaptive selection | 17.8960 | 17.1098 | 30.6816 | 466.0625 |
| PCA64 6-bit, prepared selection | 12.7237 | 12.1295 | 25.8094 | 466.0625 |

Eight-bit prepared selection reduces mean latency 35.8801% versus the prior
adaptive arm, a 1.5596x speedup without changing requested pages or extents.
Six-bit prepared selection reduces mean latency 28.9022%, a 1.4065x speedup.
Eight-bit mean latency improves in every round; round ratios are approximately
1.541, 1.627, 1.463, 1.530, 1.513, 1.661 and 1.586.

Relative to the improved UNFILTERED PREPARED baseline, the eight-bit filtered
engine is 5.2319x faster in mean latency (48.8653/9.3399). This is an internal
engine comparison, NOT a result against upstream Faiss, CLIP or DiskANN. Its
shared exact scanner is not yet an optimized SIMD production baseline.
No concurrent QPS or billion-scale performance claim is made.

## CPU and remaining costs

| Configuration | Memory-replay mean ms | Filter/coalescing mean ms, direct |
|---|---:|---:|
| Unfiltered native, prior planner | 17.3677 | 0.9404 |
| Unfiltered native, prepared planner | 16.7898 | 0.3407 |
| PCA64 8-bit, prior adaptive | 7.1128 | 6.0316 |
| PCA64 8-bit, prepared | 3.3085 | 2.1057 |
| PCA64 6-bit, prior adaptive | 8.1323 | 6.5137 |
| PCA64 6-bit, prepared | 4.1208 | 2.4036 |

The eight-bit RAM-replay execution is 2.1499x faster (53.4857% less mean wall
latency). Its real-I/O filter/coalescing category falls 65.0885%, from 6.0316 to
2.1057 ms. Six-bit RAM replay improves 1.9735x; filter/coalescing falls 63.0993%.
These are staged application measurements, not isolated arithmetic-kernel times.
Query-context setup is INCLUDED in the prepared filter timer and end-to-end time.

The eight-bit direct-I/O breakdown is 2.106 ms filter/coalescing, 5.933 ms read
stages, 1.056 ms exact-scan/top-k stages including wrapper/release costs, and
0.026 ms coarse routing. Other coordination accounts for the small remainder.
Process CPU time drops from 8.295 to 4.182 ms/query. Read-stage wall time also
falls (7.147 to 5.933 ms) despite IDENTICAL request plans. Arrival pacing and
shared-device conditions were not isolated; do not attribute that difference to
fewer reads or a separately demonstrated controller optimization.

## Invariants, memory, and verification

All 211 runner tests passed, with no errors, failures or skips. This includes
52 new prepared-path tests, existing Faiss/O_DIRECT/io_uring tests, query-context
independence, quantization-boundary masks and gap/extent-limit equivalence.
Local suite: 202 passed, nine optional dependency tests skipped. An additional
local UBSan run passed 1080 randomized decision/extent checks across six retained
dimensions (1/7/18/33/64/127) and 4/6/8-bit codes, with no sanitizer errors.

All 448 initial configuration-query comparisons matched the independent FP64
IVF candidate-set oracle. The COMPLETE staged read-plan hashes and all operation
counts agree within each precision across old/new engines. All 6272 timed queries
also returned the expected IDs/counts. There are 64 distinct queries, not 6272
independent samples. All 85268 audited rejection decisions passed against original
vectors after search; these cover the first two queries per filtered arm and
include repeated pages. No oracle radius enters online search.

All eight changed/new executable/workflow files in the downloaded source archive
match the locally tested file SHA256s. ZIP digest matches GitHub. CSV query IDs,
all operation counts, aggregate means, medians/p95/p99, memory equations and
request-plan equivalence were independently reconciled after download.

Directory arrays remain 78198802 bytes (74.576 MiB) at eight bits and 62141842
bytes (59.263 MiB) at six bits, excluding runtime/query scratch. Prepared filtered
scratch arrays at W=64 and D=128 total 3136 bytes; Python context objects and
radial caches are extra. This is not a persistent expanded-center table.
The common read pool reserved 3657728 bytes (3.488 MiB), under its 64 MiB cap.
Peak campaign RSS about 3.46 GiB includes all dataset/oracle/variant allocations
and is NOT deployment RAM.

## Environment and limits

AMD Ryzen 9 9955HX3D, 32 logical CPUs, Linux 7.0.0-34, GCC 15.2.0. Kernels use
-O3 -march=native -fno-fast-math -ffp-contract=off. Process pinned to CPU 0, not
exclusive; no fixed-frequency or device-cache control. Host load averages start
at 2.865/2.407/2.248 and end at 3.238/2.648/2.352. Existing host file locking
serializes GeoIVF speed campaigns, not unrelated work on the laptop.

O_DIRECT was actually used, but the payload fits in host RAM and the device
cache was not cold-controlled. The benchmark process retains the base/oracle.
This is one dataset/seed and reused development queries. No held-out quality,
physical NVMe-command instrumentation, production baseline, or concurrency test.
Floating-point safeguards are engineering measures, not formal numeric proofs.

The optimization is supported as a SAME-PLAN implementation improvement in this
pilot. Reader/stage pacing and native scanner efficiency are the next execution
issues; future comparisons must continue to improve the unfiltered baseline too.
Reproduction: docs/PREPARED_PLANNER.md and scripts/qualify_speed.py --prepared.
Preserve the downloaded artifact beyond GitHub's 30-day retention period.
