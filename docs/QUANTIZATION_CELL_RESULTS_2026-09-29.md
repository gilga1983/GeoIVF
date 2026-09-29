# Quantization-cell versus ball results, 2026-09-29

## Provenance

Successful tested implementation: `8156224d922e0da2cd325613f6ca12c53830ba4f`.
Workflow run: `36542000715`; job `109319396633`.
Artifact: `11020663708`, `quantization-cells-36542000715`.
Downloaded archive SHA256:
`a376bf25f20d57843043066a8344dd794be80b73bf77d165743303f4348fdf3c`.

Initial attempts stopped after the unfiltered baseline because the report builder
passed the precision field twice to dict(). That bookkeeping error was fixed;
the complete run above succeeded. A separate native revision added a genuine
fixed-work control, evaluating every active center regardless of page admission.
The result tables use only the completed revised implementation.

Canonical TexMex SIFT1M, full 1M database, unchanged 128D FP32 payloads, 4 KiB
pages, eight points/page, 125445 pages, nlist=1024, nprobe=64, k=10, seed=12345.
The same 64 development queries, frozen GeoPack pages and learning-file PCA as
preceding experiments. No held-out or multiple-dataset claim.
Payload SHA256:
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA fingerprint:
`d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Routes SHA256:
`16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.

## What changed and what did not

For each 4/6/8-bit independent PCA64 code, interpret its nearest-rounding bin
as a conservative axis-aligned cell. Store neither explicit endpoints nor a
per-point radius in the box-mode RAM object. Reuse exactly the ball code values,
origins, scales, PCA transform and page membership. Quantizer cells are padded
for engineering floating-point safeguards and checked against every indexed
projected point. Overflow fails certification rather than silently producing
an unsafe finite interval.

Ball mode keeps the measured reconstruction-error radius. Hybrid retains that
same radius array and takes max(ball lower bound, cell lower bound) PER POINT,
then min across page points. This is not the exact distance to the intersection.
The existing deterministic page-radial interval is also used in online combined
mode. It is not CLIP. Pure box RAM omits radii; the shared research input sidecar
still contains them for alternative ball loads. Memory savings refer to live
RAM arrays, not deletion of radius bytes from that shared input file.

All online comparisons use 64-page planning windows and no gap bridging.
The no-filter baseline permits whole-list reads. Online thresholds are computed
only from fetched vectors. Counts below are requested pages/extents through RAM
payload replay, not observed SSD traffic, latency, or QPS.

## Same codes, different bounding regions

All configurations return the same candidate-set answer, Recall@10=0.99375.
Unfiltered IVF requests 8394.796875 pages/query and 65.09375 extents/query.
Geometry includes actual packed codes and any retained FP32 radii. The common
8 B/page radial interval, shared quantizer/PCA and structural arrays are extra.

| Bits/coordinate | Shape | Geometry B/page | Pages/query | Page elimination | Median elimination | Extents/query |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Ball | 288 | 2189.484375 | 73.9186% | 76.5419% | 838.203125 |
| 4 | Box | 256 | 3304.000000 | 60.6423% | 62.1321% | 943.078125 |
| 4 | Hybrid | 288 | 2189.140625 | 73.9226% | 76.5474% | 838.078125 |
| 6 | Ball | 416 | 466.062500 | 94.4482% | 95.6050% | 249.968750 |
| 6 | Box | 384 | 573.328125 | 93.1704% | 94.5889% | 304.750000 |
| 6 | Hybrid | 416 | 466.062500 | 94.4482% | 95.6050% | 249.968750 |
| 8 | Ball | 544 | 297.984375 | 96.4504% | 97.0974% | 156.640625 |
| 8 | Box | 512 | 314.468750 | 96.2540% | 96.9782% | 165.500000 |
| 8 | Hybrid | 544 | 297.984375 | 96.4504% | 97.0974% | 156.640625 |

The box saves 32 geometry bytes/page: 11.11%, 7.69%, 5.88% at 4/6/8 bits.
Remaining page reads rise 50.90%, 23.02%, 5.53%, respectively. The 8-bit box is
therefore a plausible modest RAM/I/O tradeoff, not a free improvement. Keeping
both bounds adds no geometry bytes versus balls, but removes only 22 additional
page requests across all 64 queries at 4 bits, and none at 6 or 8 bits here.

The ball radius records the particular point's actual reconstruction-error norm.
The cell permits every location in the bin, including corners farther from the
center than that point. Conversely, cells constrain each coordinate separately.
Neither bound universally dominates. This workload and quantizer favor the
point-specific ball radius; the tiny hybrid gain does not justify calling the
extra computation a practical selectivity improvement in this experiment.

Complete allocated directory arrays (runtime and query scratch excluded):
4-bit ball 43.94997 MiB, box 40.12169 MiB;
6-bit ball 59.26308 MiB, box 55.43480 MiB;
8-bit ball 74.57619 MiB, box 70.74791 MiB.
Hybrid matches the corresponding ball allocation. No-filter allocation in this
research harness is not a minimal-RAM baseline.

## Native CPU mechanism experiment

Host: AMD Ryzen 9 9955HX3D, GCC 15.2.0. Generic C++17 loops compiled with -O3,
-march=native, -fno-fast-math, -ffp-contract=off. No hand-written SIMD or lookup
tables. The first 32 of the same development queries are evaluated in seven
randomly interleaved rounds, using identical candidate pages and final IVF radii.
Those final radii are ONLY offline CPU-benchmark inputs, not inputs to online
search. Both balls and boxes use squared rejection comparisons without square
roots. Every method has its decision mask checked against full-bound evaluation.

The fixed-work kernel evaluates all 64 coordinates of all active centers.
The adaptive kernel checks every 16 coordinates, rejecting centers early and
admitting a page once a completed center survives. Times include query projection,
Python/ctypes validation and calls, output allocation and checksum; they exclude
radial filtering, routing, I/O, exact vector scan and top-k. Reported values are
median wall milliseconds per query batch element, not ANN latency.

| Bits | Ball fixed | Box fixed | Hybrid fixed | Ball adaptive | Box adaptive | Hybrid adaptive |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3.11153 | 4.04225 | 4.84044 | 2.25590 | 2.38996 | 2.90415 |
| 6 | 3.18533 | 3.93991 | 4.59768 | 1.86231 | 2.25470 | 2.46683 |
| 8 | 2.43218 | 3.76552 | 4.36398 | 1.39382 | 1.83104 | 1.94475 |

Boxes are not faster in these kernels. At 8 bits, fixed-work boxes take 54.8%
more time than balls; adaptive boxes take 31.4% more. Both shapes avoid sqrt,
so radius removal does not save an otherwise necessary per-point square root.
The box adds per-coordinate absolute-value/half-bin/clamping work. This is a
plausible explanation, not a universal lower bound on optimal box CPU cost.
Different code layouts, explicit SIMD or lookup-based evaluation could differ.

There is a useful independent result: adaptive ball selection reduces this
microbenchmark's time by 27.5%, 41.5%, and 42.7% at 4/6/8 bits without changing
its selection masks. This combines partial-coordinate rejection with page early
acceptance; it is not solely the benefit of coordinate truncation. The adaptive
selection interface is not yet wired into the existing online staged executor,
which still asks for full lower bounds. Integration is a subsequent step.

The host was shared, without pinned/exclusive cores or fixed-frequency control.
Start load averages were 12.77/14.28/11.67 on 32 logical CPUs; end averages were
approximately 12.00/13.72/11.72. Raw seven-round wall and process CPU timings are
archived; no cross-host or end-to-end speedup is inferred from this experiment.

## Correctness and artifact verification

123 tests collected on the runner: 117 passed, 6 optional io_uring tests skipped.
All 18 new cell tests passed. Native direct-I/O tests ran; the unchanged async
reader retains its preceding separate validation. Local suite: 116 passed,
7 skipped, with Faiss additionally unavailable locally.

All 832 configuration-query comparisons matched independent exhaustive FP64
search over the selected IVF lists, with ID tie breaking. There are 64 distinct
queries and 13 configurations, not 832 independent samples. All 186703 audited
rejection decisions passed against original vectors; audits cover the first
two queries per configuration and include repeated pages. The three quantizer
cell constructions checked all one million indexed points each. These remain
engineering numeric guards, not formally verified floating-point arithmetic.

After download, the ZIP digest matched GitHub. Every per-query CSV ID and mean,
read_bytes=4096*read_pages, selected/gap/read relationship, CPU median and mask
checksum was reconciled. All six new code/workflow files in the runner source
archive matched their locally tested counterparts byte-for-byte. Native balls
at all three precisions exactly reproduce the legacy ball's candidate, page,
extent, stage and distance-evaluation counts, including the prior 298/466/2189
page baselines.

## Decision

Retain balls as the leading measured representation, with boxes as an explicit
radius-free RAM/I/O option rather than a claimed improvement in selectivity or
CPU. Preserve the hybrid as an ablation. The most useful overhead optimization
found here is native threshold-aware early termination with the existing balls.
Next, integrate that selection kernel into staged search and compare with real
asynchronous I/O. Neither an SSD speedup nor a CLIP advantage is established.
Reproduction details are in docs/QUANTIZATION_CELLS.md. Preserve downloaded
artifacts beyond GitHub's 30-day retention window.
