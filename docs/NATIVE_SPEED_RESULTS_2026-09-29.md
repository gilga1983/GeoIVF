# Native online speed results, 2026-09-29

## Provenance and scope

Tested implementation: `0cb7ee05709e7299c1fdc74402ad2fcb43ee8603`.
Successful workflow run: `36544239791`; job: `109326672283`.
Artifact: `11021927757`, `native-speed-36544239791`.
Downloaded ZIP SHA256:
`9e992e6bd6487ec42f3cd318402060a61233ffc15dbb33bb21ab85c84a36a890`.

Canonical SIFT1M: all 1,000,000 original 128D FP32 vectors, 125445 4 KiB pages,
nlist=1024, nprobe=64, k=10, construction seed=12345. Page membership, PCA64
transformation, independent 6/8-bit coordinate codes, and measured-error radii
are unchanged. All filtered configurations use 64-page planning windows and
zero-gap coalescing; unfiltered IVF permits whole-list extent reads.

Source GeoPack payload SHA256:
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA fingerprint:
`d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Candidate-route fingerprint:
`16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.

Correctness uses the same 64 development queries as preceding studies. Timing
uses the first 32 of those queries in five randomly interleaved rounds. These
are reused development queries, not held-out performance evidence. All methods
have Recall@10=0.99375 on the 64-query cohort and 0.996875 on the timed 32-query
subset. Do not mix page averages from the two cohorts.

Unlike earlier RAM-only page studies, this campaign actually executed online
searches with asynchronous O_DIRECT reads through the existing io_uring reader.
These are application end-to-end timings, not MQSim predictions or sums of I/O
latencies. The comparison is against unfiltered IVF in OUR engine with the SAME
native FP64 scan and reader, not upstream Faiss, CLIP, or DiskANN performance.

## Implemented changes

- Integrate the native threshold-aware selector into the staged executor. It
  uses only the current kth distance from already fetched original vectors.
  Radial page rejection precedes the ball kernel. The kernel can reject a ball
  after a partial coordinate sum, or accept a page after one complete ball survives.
- Add a shared native exact-distance/top-k kernel: FP32 input bytes, FP64 L2
  accumulation, bounded max heap and ID tie-breaking, one call per completed stage.
  Both filtered and unfiltered baselines use this implementation.
- Add bounded reusable aligned mmap buffers. The native scan borrows memoryviews
  until the stage is consumed instead of copying returned bytes. A new read is
  forbidden until the previous batch is released. Device completion and canonical
  stage processing remain unchanged; no asynchronous or direct-I/O fallback occurs.
- Expose --summary, --selection, --scan and --pooled options in the existing search
  CLI. Old defaults remain compatible and the new paths are explicitly selected.

## Canonical correctness and operation counts

All 159 runner tests passed with no failures or skips, including Faiss,
O_DIRECT, io_uring, pooled-buffer lifetime/error checks and 36 new speed tests.
Local development validation: 150 passed, 9 optional dependency tests skipped.

All 512 initial configuration-query comparisons (eight configurations on 64
queries) matched the independent FP64 exhaustive candidate-set answer. At each
coordinate precision, the complete staged request-plan SHA256 and every page,
byte, extent, stage and exact-distance count agreed across the compared engines.
102462 rejected-page decisions were audited after search against original vectors;
all passed. Audits cover the first two queries of each filtered configuration and
include repeated pages. They are not distinct page/query counts.

All 2240 timed searches also returned the reference IDs and the same expected
operation counts. Timing checks occur outside the timed interval. Every timed
search performs a fresh Faiss routing call and resets any cached projection;
subsequent thresholds are always derived online. No final-radius oracle is used.

| Setting, 64-query cohort | Pages/query | Extents/query |
|---|---:|---:|
| Unfiltered, Python or native | 8394.796875 | 65.093750 |
| PCA64 8-bit, legacy/full/adaptive | 297.984375 | 156.640625 |
| PCA64 6-bit, full/adaptive | 466.062500 | 249.968750 |

Thus this milestone preserves the preceding pruning result. It changes execution
cost, not summary precision, candidate lists, or output recall.

## Actual asynchronous direct-I/O timings

Mean/median/p95 are wall milliseconds per complete query, including coarse
routing, projection, filtering/coalescing, stage coordination, reads, exact scans
and top-k. The reader is opened with O_DIRECT and io_uring at queue depth 16.
The same 32 queries run five times per arm; each timing row below has 160 samples
but only 32 distinct queries. Process affinity is pinned to CPU 0, not exclusive.

| Engine | Reader | Mean ms | Median ms | p95 ms | Pages/query | Extents/query |
|---|---|---:|---:|---:|---:|---:|
| Unfiltered native IVF | Pooled | 51.9299 | 52.6520 | 60.7942 | 8531.156250 | 65.250000 |
| PCA64 8-bit, full native bounds | Pooled | 14.1713 | 14.1209 | 20.4566 | 295.562500 | 151.406250 |
| PCA64 8-bit, adaptive native selection | Pooled | 14.1456 | 14.4291 | 21.4298 | 295.562500 | 151.406250 |
| PCA64 6-bit, adaptive native selection | Pooled | 17.9523 | 18.8289 | 26.2153 | 459.500000 | 240.343750 |
| Unfiltered native IVF | Allocating/copying | 62.3341 | 63.0341 | 68.6273 | 8531.156250 | 65.250000 |
| PCA64 8-bit, adaptive native selection | Allocating/copying | 15.4168 | 15.9241 | 23.5565 | 295.562500 | 151.406250 |

On the common pooled/native stack, eight-bit filtering improves mean latency
by 3.6711x (72.7602% lower mean latency) relative to unfiltered IVF in this engine.
Per-round ratios of mean latency are 3.672, 3.698, 3.676, 3.654, and 3.657.
That internal repeatability is not a multi-host or held-out confidence interval.

Pooled buffers reduce eight-bit adaptive mean latency from 15.4168 to 14.1456 ms,
approximately 8.25%. The same change helps the unfiltered baseline more, from
62.3341 to 51.9299 ms, approximately 16.69%. The main comparison therefore uses
pooled buffers on BOTH sides rather than inflating gains with a copying baseline.

The additional adaptive-versus-full-bound benefit is negligible in this first
online implementation: 14.1713 versus 14.1456 ms in the direct-I/O means, with
mixed medians/tails. Do not claim the earlier 43% isolated kernel improvement
translated to a 43% online speedup. Per-window Python validation, allocation and
native-call overhead, plus I/O, limit that benefit in this integration.

## CPU-only execution diagnostics

MemoryReplay removes device waits, but still executes real staged search over
FP32 payload bytes. It is not an optimized in-memory Faiss baseline.

| Configuration | Mean wall ms | Median wall ms |
|---|---:|---:|
| Unfiltered, Python scan | 40.0823 | 39.7034 |
| Unfiltered, native scan | 18.1562 | 18.1822 |
| Legacy NumPy/unpacked ball filter + Python scan, 8-bit | 81.0038 | 80.8450 |
| Full native ball bounds + Python scan, 8-bit | 7.9685 | 7.5706 |
| Full native ball bounds + native scan, 8-bit | 7.5808 | 7.3238 |
| Adaptive native selection + native scan, 8-bit | 7.4832 | 7.3484 |
| Full native bounds + native scan, 6-bit | 9.0358 | 8.5447 |
| Adaptive native selection + native scan, 6-bit | 8.5661 | 8.2617 |

The large reduction from the legacy eight-bit implementation is mostly removal
of generic NumPy bit unpacking/expanded-center arrays, not stronger geometry.
Native scanning also improves the unfiltered baseline. Since few vectors remain
under the strong filter, its scan-only improvement is naturally smaller.

The direct pooled eight-bit adaptive breakdown averages 5.946 ms in filtering
and coalescing, 6.774 ms in the read stage, 1.098 ms in exact-distance/top-k stage
(including wrapper/buffer-release overhead), and 0.027 ms in coarse routing.
Unfiltered pooled native execution spends 34.847 ms in read stages and 15.906 ms
in distance/top-k stages. Remaining time includes other coordinator overhead.
These are application timing categories, not isolated device service times.

## Memory and environment

Geometry stays 544 B/page at eight bits and 416 B/page at six bits, plus the common
8-byte radial interval. Complete directory arrays are 78198802 bytes (74.576 MiB)
and 62141842 bytes (59.263 MiB), respectively. Runtime and per-query scratch are extra.
The borrowed read pool reserved 3461120 bytes (3.301 MiB) in this campaign, under
its 64 MiB cap, with 59 growth/allocation events across the campaign. The native
heap uses O(k) per-query storage. No expanded full-index FP64 center cache is kept.

Host: AMD Ryzen 9 9955HX3D, 32 logical CPUs; compiler GCC 15.2.0; Linux 7.0.0-34.
Kernels compile with -O3 -march=native -fno-fast-math -ffp-contract=off.
No hand-written SIMD distance reduction is implemented yet. Host load averages
start at 2.77/2.53/4.80 and end at 2.69/2.57/4.67. One CPU was pinned, but other
workloads, frequency scaling and device caching were not controlled. A file lock
serializes GeoIVF speed campaigns only, not other repositories' jobs.

The 513822720-byte payload fits in host RAM. O_DIRECT bypasses the host page cache,
but this is not an out-of-memory-scale or cold-device experiment. The database and
oracle also remain resident in the benchmark process. Peak experiment RSS (about
3.46 GiB) must not be quoted as deployed index RAM. Application requested extents
are not independently instrumented physical NVMe commands or controller traffic.

## Independent verification and next bottleneck

After artifact download, ZIP SHA256 matched GitHub. All eleven changed/new source
files in the runner source.zip matched the locally tested fingerprints. Every
CSV query identity and aggregate mean, timing median/percentile, round mean,
read_bytes=4096*read_pages and selected/gap/read-page equation was reconciled.

The first real-I/O pilot is positive for the PCA filtering layer. It is not a
3.67x claim against upstream Faiss/CLIP, a concurrency/QPS result, or a production
or multi-dataset result. FP64 sum order can differ from NumPy at numerical ties;
integer-coordinate SIFT answers/plans are checked exactly, while noninteger test
distances use a stated tolerance. Conservative guards are not a formally verified
floating-point library.

The remaining CPU bottleneck is repeated per-window filter/coalescing machinery.
A prepared query context, fused native selection and a stronger native/SIMD common
scan are sensible next engineering steps. Real benchmark expansion still needs
upstream CLIP, held-out queries, multiple datasets and a controlled storage setup.
Reproduction and CLI details: docs/NATIVE_SPEED.md. Preserve the downloadable
artifact beyond GitHub's 30-day retention period.
