# SIMD exact scanning and rolling asynchronous reads

These paths are opt-in. The existing scalar scanner and queue-depth-batched
reader remain available as matched controls. Page layout, PCA64 codes, radii,
and the prepared conservative filter are unchanged.

## Exact scanner

`native/simd_topk.cpp` reuses the scalar accumulator's heap, constructor and
output/error ABI. It adds an AVX2 scanner that evaluates four DIFFERENT points
in parallel. Four adjacent FP32 coordinates from each point are loaded without
alignment/aliasing assumptions and transposed. Each SIMD lane belongs to one
point and accumulates FP64 terms in the same coordinate order as the scalar
scanner. It does not horizontally reduce dimensions, use fast-math, contract
multiply/add into FMA, or reduce precision. Heap insertion order is unchanged.

Partial pages, fewer than four remaining points and non-multiple-of-four
coordinate dimensions use scalar tails. The explicit SIMD path checks AVX2 and
fails clearly on unsupported hosts instead of reporting a silent fallback as
SIMD. The original `--scan native` path remains portable to its build target.
`--scan simd` selects the new implementation. No transposed copy of the database
or persistent expanded representation is stored; transposition uses registers.

## Rolling reader

`native/rolling_reader.cpp` preserves the completed-stage API and the existing
pool's buffer lifetime. Unlike the previous reader, it does not wait for every
request in a queue-depth batch before replenishing available slots. It drains
ready completion events and fills freed slots, keeping at most the configured
queue depth in flight. Offsets, request lengths, and output-buffer association
are unchanged. Every submitted request has an explicit completion ID. Invalid,
duplicate or short completions are rejected. On an ordinary I/O error, no new
requests are submitted and outstanding requests are drained before returning
failure. Exceptional queue failures poison and close the ring, matching the
existing reader's failed-state contract. The pool remains owned until the C
call returns; output views cannot be reused until the completed stage is consumed.

This does NOT overlap the next search planning window with unfinished reads.
The kth threshold is updated only after the same canonical completed stage,
so changing batch to rolling I/O does not change the algorithmic read plan.
Rolling completion bookkeeping costs one transient byte per stage request;
there is no new per-vector persistent metadata.

Build: `bash scripts/build_execution.sh --uring`.
CLI: `--selection prepared --scan simd --backend uring --direct --pooled`.
Add `--io-schedule rolling` to select rolling I/O; `batch` remains the default.

## Canonical campaign

`scripts/qualify_execution.py` retains the previous SIFT1M dataset, learning-file
PCA, IVF centroids/assignments, frozen GeoPack pages and development-query split.
It runs 64 queries in five randomly interleaved rounds for each of two phases:
RAM replay and actual pooled O_DIRECT/io_uring execution at queue depth 16.

The first comparisons are SAME-PLAN engineering controls: scalar versus SIMD,
and batch versus rolling, under identical precision/window/gap parameters.
Unfiltered IVF receives the new scanner and reader too and can use whole-list
extents. Complete stage/extent plan hashes and all operation counts must agree
within each such group, in addition to the independent FP64 candidate-set answer.

A separate scheduling sweep uses 16, 64, 256 and whole-list windows plus a
256-page, two-page-gap option. These intentionally change the read plan. They
must preserve answers, but are NOT advertised as equal-page-count optimizations.
All thresholds originate from fetched vectors, never the final answer oracle.

Every timed search performs fresh coarse routing, query preparation, actual
stage reads, exact scan and top-k. Assertions/checksums run after timing. The
host file lock serializes GeoIVF speed campaigns only; CPU affinity is not
exclusive and does not control device cache or frequency. This is a single-
dataset development experiment, not held-out, out-of-RAM-scale, concurrent-QPS,
physical-NVMe-command instrumentation, or a Faiss/CLIP/DiskANN comparison.

New tests cover exact scalar/SIMD FP64 equality on noninteger data, different
coordinate dimensions and k, ties, unaligned buffers, partial pages, invalid
payload, full-search read-plan equivalence and rolling reads at depths 1/2/8/32
with both direct and buffered I/O. Actual optional-dependency execution and
measured results are recorded separately after the runner completes.
