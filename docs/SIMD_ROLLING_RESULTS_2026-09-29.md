# SIMD scanner and rolling I/O results, 2026-09-29

## Provenance and scope

Tested implementation: `079192799878531e3b234f191952544502f12fbd`.
Successful workflow: `36577616088`; job: `109437175979`.
Artifact: `11037524261`, `execution-speed-36577616088`.
Downloaded ZIP SHA256:
`f496da5ac07508b41d92596dd1d36e4c11e381fc3ac95b37c9790daac9562dfd`.

Canonical SIFT1M: 1M unchanged 128D FP32 vectors, 125445 4 KiB pages, eight
vectors/page, nlist=1024, nprobe=64, k=10, seed=12345. Same 64 DEVELOPMENT queries,
learning-file PCA64, frozen GeoPack pages and independent 6/8-bit balls as before.
All twelve arms return the same independent exhaustive IVF candidate-set answer;
Recall@10 is 0.99375 for every arm. Timings use five randomly interleaved rounds
per arm in each phase, giving 320 samples/arm but only 64 distinct queries.

Payload: `8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA: `d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Routes: `16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.
These fingerprints match prior canonical studies. Comparisons below rerun old
and new methods TOGETHER; they do not mix timings from previous sessions.

## Implemented changes

1. An explicit AVX2 exact scanner vectorizes across four POINTS, retaining
   sequential FP64 coordinate addition within each lane. A register transpose
   loads the unchanged FP32 point-major payload. No horizontal sum reorder,
   reduced precision, FMA contraction or fast-math. Heap insertion order and
   ID tie-breaking remain unchanged. Unaligned buffers and scalar tails work.
2. A rolling io_uring reader replenishes freed queue slots within each stage
   rather than waiting for a full queue-depth group. It preserves request
   offsets/sizes, completion-buffer IDs and completed-stage semantics. Short,
   invalid and failed completions poison the reader; ordinary I/O errors drain
   outstanding requests. The old reader remains an explicit control.
3. A SEPARATE scheduling sweep changes windows and gap bridging, keeping
   precision and geometry fixed. These changes preserve answers but do NOT
   preserve read plans. All thresholds still originate from fetched vectors.

The prepared planner and pooled buffers are used in every arm, including
unfiltered IVF. Direct measurements use actual O_DIRECT/io_uring with queue depth
16. SIMD is opt-in through --scan simd; rolling through --io-schedule rolling
with --backend uring --pooled. Existing defaults and comparison paths are retained.

## Complete-query direct-I/O measurements

Milliseconds including fresh routing, prepared filtering, all stage reads,
exact scans and top-k. These are application timings, not isolated device service
latencies, a Faiss/CLIP comparison, or concurrent query throughput.

| Configuration | Mean ms | Median ms | p95 ms | Pages/query | Extents/query | Stages/query |
|---|---:|---:|---:|---:|---:|---:|
| Unfiltered, scalar scan, batch reader | 49.1054 | 48.8515 | 57.4293 | 8394.7969 | 65.0938 | 64.0000 |
| Unfiltered, SIMD scan, batch reader | 22.0879 | 22.0177 | 24.4848 | 8394.7969 | 65.0938 | 64.0000 |
| Unfiltered, SIMD scan, rolling reader | 22.2321 | 22.1098 | 24.5876 | 8394.7969 | 65.0938 | 64.0000 |
| 8-bit, scalar scan, batch, window64 | 9.1161 | 8.3817 | 16.0574 | 297.9844 | 156.6406 | 47.9375 |
| 8-bit, SIMD scan, batch, window64 | 8.7132 | 8.0212 | 17.7773 | 297.9844 | 156.6406 | 47.9375 |
| 8-bit, SIMD scan, rolling, window64 | 8.6844 | 8.0476 | 15.1511 | 297.9844 | 156.6406 | 47.9375 |
| 6-bit, scalar scan, batch, window64 | 12.2725 | 11.6661 | 23.3849 | 466.0625 | 249.9688 | 65.3125 |
| 6-bit, SIMD scan, rolling, window64 | 11.6992 | 11.3128 | 21.3418 | 466.0625 | 249.9688 | 65.3125 |
| 8-bit, SIMD scan, rolling, window16 | 14.2081 | 12.8425 | 25.0514 | 279.2656 | 166.8281 | 92.3281 |
| 8-bit, SIMD scan, rolling, window256 | 6.4645 | 6.2544 | 10.9450 | 335.6875 | 146.7188 | 27.5938 |
| 8-bit, SIMD scan, rolling, whole-list window | 6.3836 | 6.0602 | 12.3165 | 336.1250 | 146.6406 | 27.4375 |
| 8-bit, SIMD scan, rolling, window256, gap2 | 6.0696 | 5.9007 | 10.3008 | 394.2656 | 103.9375 | 27.5938 |

The fastest tested arm is window256/gap2. Against the previous scalar/batch
prepared implementation rerun here, mean latency improves 1.5019x (33.4183%
lower), 9.1161 to 6.0696 ms. The five round ratios are 1.481, 1.498, 1.484,
1.473 and 1.574. This is repeated development evidence, not held-out validation
of a selected configuration or a cross-host confidence interval.

The stronger unfiltered baseline matters. SIMD/batch reduces unfiltered mean
latency 55.0194%, from 49.1054 to 22.0879 ms. The fair best-tested comparison is
therefore 22.0879/6.0696 = 3.6391x, not the old 5.23x headline against the weaker
scalar scanner. Under the exact same rolling reader on both sides the ratio
is 3.6628x. These are comparisons within our engine, not upstream systems.

## Where the improvements come from

SIMD is a large common-scanner improvement but only a small complete-query
improvement once pruning already avoids most scans. On 8-bit window64, changing
only scalar to SIMD improves mean direct latency 4.4188%. Round ratios and p95
are mixed, so do not present this as a uniform tail-latency improvement.
The exact-scan stage decreases from 1.0541 to 0.6126 ms, including wrapper costs.

Rolling versus batch under the same SIMD/window64 plan changes mean latency
only from 8.7132 to 8.6844 ms (0.33%). Most individual round ratios do not favor
rolling; there is NO convincing independent reader win in this campaign.
Unfiltered rolling is slightly slower than unfiltered batch. Preserve rolling
as an available scheduling alternative, not a proven universal optimization.

The clear win is fewer serial planning/read stages. Window64 to window256 with
SIMD/rolling reduces mean latency 25.5616% while increasing pages from 297.9844
to 335.6875. Adding gap2 reduces latency another 6.1089% while increasing pages
to 394.2656. Gap bridging deliberately fetches some rejected pages. The fastest
arm has 32.3108% MORE pages than the original 8-bit plan, but 42.4381% fewer read
stages and 33.6459% fewer extents. It still eliminates 95.3035% of unfiltered
candidate-page requests. Requested bytes are 1.5401 MiB/query versus 32.7922 MiB.

Window16 illustrates the opposite tradeoff: it minimizes pages among tested
arms, but 92.33 stages/query make it much slower (14.21 ms). Minimum page count
is not minimum wall-clock latency. The gap2 setting was not separately paired
with a batch reader in this sweep, so do not infer rolling is necessary for its
benefit. This is a small scheduling sweep, not a global configuration optimum.

## CPU diagnostics and remaining cost

| Configuration | RAM-replay mean ms | Direct mean process CPU ms |
|---|---:|---:|
| Unfiltered scalar/batch | 16.7222 | 17.6841 |
| Unfiltered SIMD/batch | 2.6979 | 4.7871 |
| 8-bit scalar/batch/window64 | 3.2786 | 4.1294 |
| 8-bit SIMD/batch/window64 | 2.7752 | 3.6955 |
| 8-bit SIMD/rolling/window256/gap2 | 2.1942 | 2.8869 |

The unfiltered RAM-replay scan/top-k category drops 15.4903 to 1.4816 ms.
In direct I/O it drops 15.7225 to 2.8324 ms. Read-stage wall time ALSO drops
32.8335 to 18.7060 ms despite identical requests. Arrival pacing and CPU/device
cache interaction were not isolated; do not attribute that part to fewer reads
or a separately established controller benefit.

The fastest filtered direct arm spends 1.6919 ms in filter/coalescing, 3.7393 ms
in read stages and 0.4582 ms in scan/top-k, plus routing/coordination. Its CPU
work and storage waits remain both relevant. RAM-replay is our staged harness,
not a comparison against optimized in-memory Faiss.

## Correctness, memory and independent verification

All 260 runner tests passed: zero failures, errors or skips. This includes 49
new scanner/reader tests and the previous Faiss, O_DIRECT and io_uring tests.
Rolling reads were tested at depths 1/2/8/32, buffered and direct, with varying
sizes, shuffled and duplicate addresses, more requests than queue depth, and
mixed successful/failed reads. SIMD tests require BITWISE scalar/SIMD distance
agreement on noninteger data, multiple dimensions/k, ties and unaligned tails.
Local suite: 241 passed, 19 optional dependencies skipped. An additional local
UBSan test passed 100 randomized scalar/SIMD comparisons across 300 stages.

All 768 initial configuration-query comparisons matched the independent FP64
candidate-set oracle. Complete staged-plan hashes and all operation counts agree
within each fixed precision/window/gap group. All 7680 timed searches match their
expected IDs and operation counts. There are only 64 distinct queries, not 7680
independent samples. 153741 rejected-page decisions were audited afterward, all
passing; audits reuse first-two-query decisions across configurations.

After downloading the artifact, every CSV identity/count, aggregate mean,
median/p95, per-round mean, byte/page equation and same-plan-group hash was
independently reconciled. All nine executable/workflow source files match the
locally tested versions byte for byte. The ZIP digest matches GitHub's digest.

Persistent directories are unchanged: 78198802 bytes (74.576 MiB) at eight bits,
62141842 bytes (59.263 MiB) at six bits, excluding runtime and query scratch.
Geometry remains 544/416 B per page plus the common radial interval. The shared
batch pool reserved 3.488 MiB; the rolling pool reserved 8.254 MiB across its
multiple window/gap arms, each under its 64 MiB cap. These are campaign peaks,
not separately measured per-configuration steady-state minima. With W=256,
prepared scratch arrays are 6400 bytes plus Python objects/cache/extent lists.
Peak campaign RSS (about 3.46 GiB) includes database, oracle and all variants.

## Limits and decision

AMD Ryzen 9 9955HX3D, 32 logical CPUs, Linux 7.0.0-34; process pinned to CPU 0,
not exclusive. Host load starts 2.900/2.386/2.273 and ends 2.583/2.415/2.291.
No fixed-frequency or cold-device control. O_DIRECT is genuine, but the small
payload fits host RAM and database/oracle remain in the benchmark process.
No concurrency/QPS, instrumented physical NVMe command, held-out, multi-dataset,
out-of-RAM-scale or CLIP result is established. Numeric guards are engineering
safeguards, not a formally verified floating-point library.

Carry SIMD forward as the stronger common scanner. Use window256/gap2 as the
fastest measured DEVELOPMENT configuration, retaining the no-gap alternative
for lower traffic. Do not claim an independent rolling-reader speedup. Next
comparison should validate the selected schedule on held-out/larger workloads
and anchor execution against upstream baselines before further headline claims.
Reproduction: docs/SIMD_ROLLING.md and scripts/qualify_execution.py. Retain the
artifact beyond GitHub's 30-day retention period.
