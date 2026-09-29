# First released-system comparisons

These campaigns replace the previous same-engine-only speed baseline with
public upstream algorithms. They are two DIFFERENT scoreboards, not one mixed
RAM/SSD latency leaderboard. No SOTA superiority is established by launching
or compiling an experiment; results must be interpreted after actual execution.

## Common data and split

Canonical TexMex SIFT1M, unchanged 1M 128D FP32 base vectors and supplied ground
truth. The base file SHA256 is
21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816.
All previous tuning used the first 1000 elements of a query permutation seeded
with 20260929. These campaigns use the first 128 of that development pool for
calibration, and permutation positions 1000:1256 as 256 held-out queries.
The partitions are disjoint. Search parameters are recorded/frozen BEFORE any
held-out measurement and are not changed after seeing held-out recall.

The development target is mean Recall@10 >= 0.99. Each algorithm chooses the
fastest tested configuration reaching it. This does not guarantee identical
held-out recall or that every held-out result exceeds 0.99. Report the actual
held-out values and do not relabel a missed target as matched-recall success.

GeoIVF keeps its previously selected PCA64 independent eight-bit balls,
prepared native filtering, same-order FP64 SIMD scan, window256 and gap2.
Global PCA is fitted on sift_learn.fvecs, not calibration or held-out queries.
Coarse IVF training is seed12345, 1024 lists, 100000 sampled base vectors.
Original within-list GeoPack remains the frozen 16-coordinate packing.
No representation/layout parameters are fitted to held-out queries.

## Released IVF scoreboard: all vectors in RAM

Pinned SongYitong826/CLIP commit:
7f4fc84edffede0aa21fae6131ec7391ce99ab6f.
The upstream Faiss submodule and all source files remain unchanged. A small
measurement bridge calls IndexIVFFlat, IndexIVFCLIP, and IndexHIVFCLIP directly.
Build glue supplies OpenMP/BLAS linkage and position-independent compilation.
The upstream release optimization flags remain in use. BLAS packages are
extracted into the job directory, not installed with sudo.

All three upstream algorithms receive the SAME coarse centroids as GeoIVF.
Their released add/search/hierarchy routines perform actual indexing/search.
IVF-CLIP lambda settings match the small-nlist released example: N100,
1000 sampled queries, 1000 sampled objects, quantile0.9999. HIVF uses its example
leaf settings N4, n_q3000, n_c3000, quantile0.999, and height2 upper settings
n_q30000, n_c500, N5, quantile0.99999. Calibration samples for learned pruning
come from base vectors, not the held-out query file. This is a common-centroid
controlled configuration, not an exhaustive reproduction of all paper settings.
The stock packaged Faiss version is an additional control and is recorded.

Tune nprobe over 8/16/32/64/96/128/192/256 on the development set. Five rounds
interleave algorithm and query order on the held-out set. All run single-query
search with one OpenMP thread on one pinned CPU. GeoIVF uses MemoryReplay for
this scoreboard, so no SSD waits are charged to one method but not another.
Its Python stage coordinator is included, as are external ctypes/Python call
costs. Upstream numerical kernels use FP32/optimized math; Geo uses conservative
FP64 bounds and same-order FP64 exact scanning. The separate fixed-nprobe64
check reports ordered-ID and set agreement with a candidate-set FP64 oracle;
tie-order differences must not be presented as lost neighbors without diagnosis.

These all-resident timings are CPU/execution context, NOT the RAM-budgeted disk
system claim. Resident upstream vector payloads occupy roughly 512 MB before
IDs/index metadata; Geo MemoryReplay also retains its payload for this diagnostic.
Multiple indexes coexist in the process, so total RSS is not any system's
minimal deployment memory. Quantile learned pruning does not inherit GeoIVF's
zero-additional-loss certificate merely by being compared with it.

## Released DiskANN3 scoreboard: actual disk search

Pinned microsoft/DiskANN main commit:
fcf90534174cf29c78c9f13b4cccf1fcabff85f5, Rust toolchain1.97.1.
This is the public DiskANN3 implementation, not the legacy C++ cpp_main branch.
Build the official diskann-benchmark with the disk-index feature, --locked and
-C target-cpu=native. No upstream search or training code is modified.
Its Linux aligned reader opens the graph/payload with O_DIRECT.

Build R64/L100, four build threads, FP full-vector graph construction, 64 PQ
chunks for resident navigation codes, 6 GB build-memory allowance. Search is
single-thread graph mode, not flat scan, with no node cache and no I/O limit.
Important API detail: num_nodes_to_cache=null selects CachingStrategy::None;
zero is rejected by the public input validator. This is not a nonzero cache.
Tune search_list over 20/40/60/80/120/200/400 and beam widths4/8. Geo nprobe grid
is 8/16/32/64/96/128. Freeze both at the 99% development recall target, then run
three alternating system-block rounds over the same 256 held-out queries.

PQ64 costs 64 code bytes/vector but is NOT all of DiskANN's query memory. Geo
has its existing roughly74.6MiB directory plus bounded I/O buffers and runtime.
No equal hard-RSS/cgroup limit is imposed in this first integration experiment.
Report index file sizes and explicit cache/code settings; do not claim exact
matched-memory domination or compare peak build RSS with query directory bytes.
The DiskANN build/initial calibration runs before process pinning; both held-out
systems inherit the same single-core affinity. The graph constructor remains
free to build with its declared four threads.

Official DiskANN statistics report mean_latency in microseconds, qps, I/O
operations, CPU time and recall in percent. Its timer is the searcher's internal
execution timer. Geo timings include routing and Python stage coordination.
1/QPS provides a separate amortized upstream wall-time indication but is not
concurrent service throughput here. The official recall metric handles ties
using distances; Geo reports strict supplied top10 ID overlap. Small differences
must be qualified. No final-radius oracle drives either online algorithm.

## Validation and limits

Both workflows preserve raw configs, selected parameters, source/revision
information, per-query Geo/IVF results, official DiskANN outputs and logs.
Run existing native/O_DIRECT/io_uring tests for the disk campaign; the all-RAM
campaign still builds the base reader library for existing regression tests.
The host file lock serializes these benchmark processes and previous GeoIVF
speed campaigns, not unrelated jobs or dependency compilation. CPU affinity
is not exclusive and does not freeze clocks or device caches.

This is one dataset, one construction seed, one small held-out cohort, small
payloads that fit host RAM, and no independently instrumented NVMe commands.
Actual O_DIRECT is not proof of cold-device or out-of-RAM-scale behavior.
No PageANN/SPANN/TRIM/RaBitQ performance result is included in these scripts.
Use this milestone to locate the performance gap and select the next fair
comparison, not to claim that one measured configuration settles all SOTA.
