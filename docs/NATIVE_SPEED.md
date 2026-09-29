# Native online execution qualification

This milestone optimizes execution rather than compression. Frozen GeoPack pages,
PCA64 codes, FP32 payloads, radii, IVF candidate lists and stage windows are unchanged.
The implementations are optional; existing `search` defaults remain compatible.

## Changes

1. `CellIndex.select` accepts the actual online kth threshold and calls the existing
   native fixed/adaptive selector. It can reject a ball after partial-coordinate
   accumulation and can stop scanning a page after one completed ball survives.
   The deterministic radial page bound is applied before geometry. Infinite initial
   thresholds retain seed pages. No ground-truth or final radius enters online search.
2. `NativeTopK` scans original FP32 bytes, accumulates L2 squared distances in FP64,
   and maintains a bounded max heap with ID tie-breaking. It receives all completed
   extents in canonical order through one call per stage. The same kernel serves
   filtered and unfiltered IVF. No quantized distances determine the output ranking.
3. `PooledNative` reuses bounded aligned mmap buffers and returns borrowed memoryviews,
   avoiding the prior allocation/read/copy/free cycle. Views are released after a
   complete stage is consumed; another read while views are outstanding is an error.
   Buffers remain live throughout device completion and scanning. No direct-I/O or
   asynchronous request silently falls back to buffered or synchronous execution.

Build (no sudo):

```bash
bash scripts/build_speed.sh
bash scripts/setup_uring.sh
GEOIVF_TEST_DIRECT=1 GEOIVF_TEST_URING=1 python -m pytest -q
python scripts/qualify_speed.py --work /tmp/geoivf-speed-unique \
  --out artifacts/speed-unique --queries 64 --timed-queries 32 --repeats 5 --with-direct
```

The CLI can use an existing certified independent PCA sidecar:

```bash
python -m geoivf search --index data/frozen-pages --summary data/pca64-b8 \
  --queries data/sift_query.fvecs --out artifacts/online-speed \
  --backend uring --direct --pooled --selection adaptive --scan native \
  --filter combined --nprobe 64 --window-pages 64 --queue-depth 16
```

Sidecar construction and certification are performed by the qualification scripts.
The historical generic `build` CLI still constructs its legacy coordinate summary;
it does not silently switch to PCA or generate a certified sidecar.

## Evaluation contract

The workflow runs only on a trusted experiment branch or manual main dispatch,
with read-only repository permissions and one job. It serializes this speed campaign,
uses the shared dataset cache, and deletes only its own temporary payloads.

First verify all original 64 development queries. Compare the complete staged
request-plan SHA256 as well as pages, bytes, extents, stages, distances, and neighbor
IDs for the legacy Python and new native variants at each precision. Recompute
rejected numerical bounds only in this correctness phase; audit vectors after search.
Timed queries do not collect such audits or traces.

Then execute 32 of these same queries in five randomly interleaved rounds. Every
search resets its projection cache before timing, includes a fresh Faiss coarse
routing call, and derives subsequent thresholds from fetched payloads only.
MemoryReplay isolates implementation overhead. The optional storage run actually
uses `O_DIRECT` plus io_uring at depth 16. Both the unfiltered and filtered comparisons
use the identical native scan and pooled reader. Additional copying-reader controls
isolate buffer reuse. Every timed answer and operation count is checked outside its
timed interval. Warmup runs exercise all arms before measurements.

The timing process is pinned to one allowed CPU. This is NOT core reservation or
host isolation. A host-local file lock coordinates GeoIVF speed campaigns, not work
from other repositories. Host load, hardware, affinity, package versions and source
are archived. No system cache flushing, service changes, or root operations occur.

## Numerical and resource limits

Compilation disables fast-math and fused FP contraction. The original finite input
range and conservative ball guards remain in force. Native sum order can differ
from NumPy near floating-point ties, so arbitrary-data bitwise equality is not
claimed. Integer-coordinate SIFT output IDs and plans are required to match exactly;
noninteger regression distances are checked with a declared tolerance.

The native heap uses O(k) memory per query. Its byte inputs are borrowed, not copied.
The read pool has a 64 MiB ceiling, with actual reservation/allocation counts reported.
Reported directory arrays exclude process overhead and transient scratch. Experiment
RSS also includes the database and exhaustive oracle, and is NOT deployment RAM.

Application reads are not instrumented physical NVMe commands. Direct I/O avoids
the host page cache but device caches can remain warm. SIFT1M fits in host RAM; no
large-scale/out-of-RAM claim, CLIP comparison, concurrent QPS claim, or production
latency conclusion follows. Python still coordinates stages, the reader waits for
whole completion stages, and no hand-written SIMD scan is included yet.

Buffer-lifetime reference: https://man7.org/linux/man-pages/man7/io_uring.7.html
