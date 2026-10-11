# GeoIVF

Research infrastructure for page-oriented execution of disk-resident IVF.

```
Faiss training and coarse routing
    -> input / radial / geometric within-list layout
    -> staged conservative page planner
    -> coalesced byte extents
    -> memory replay | buffered pread | native pread/O_DIRECT | liburing
                              |
                              +-> dependency-bearing JSONL -> MQSim open-loop export
```

FP32 payload vectors remain unchanged in the page file. Only the RAM summaries
are quantized. The native adapter uses POSIX and upstream liburing; this project
does not implement an SSD model. MQSim is an external, commit-pinned dependency.

## Status and scope

This is an executable evaluation foundation, not a performance result or a
production search engine. The coordinator and exact distance loop currently
run in Python/NumPy. Native readers include buffer allocation/copying costs.
CLIP/HIVF-CLIP, large-scale canonical datasets, PCA projection, multi-query
scheduling, and calibrated NVMe measurements are not integrated yet.

The `radial` filter is the deterministic reverse-triangle inequality over a
page's centroid-distance interval. It is NOT CLIP's learned angular bound and
must never be labeled CLIP in a results table. `geopack` is currently balanced
recursive coordinate partitioning within existing IVF lists, not a claim of a
new clustering algorithm.

Initial local validation: 30 tests passed with direct-I/O tests enabled. Faiss
and io_uring execution were skipped because their dependencies were unavailable
in the isolated development container. The integration workflow requires Faiss
and runs upstream MQSim on a small synthetic fixture. Check its actual status
before treating either integration as validated.

## Setup

Linux, Python 3.10+, a C++17 compiler, and `make`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make
GEOIVF_TEST_DIRECT=1 python -m pytest -q
python scripts/smoke.py --out artifacts/smoke
```

Dependencies have bounded compatible ranges. Every CI run archives `pip freeze`
and its Git commit. Freeze those resolved versions before a performance campaign.

For the optional asynchronous backend, install the distribution's `liburing-dev`
package, then explicitly rebuild and validate it:

```bash
make clean
make URING=1
GEOIVF_TEST_DIRECT=1 GEOIVF_TEST_URING=1 python -m pytest -q
```

A disabled kernel interface, missing dependency, alignment failure, or failed
read causes an error. No backend silently falls back from direct to cached I/O
or from io_uring to synchronous reads.

## Build and search

Input: finite float32 `.npy` or `.fvecs`, L2 metric, vector magnitudes <= 1e10.
Construction currently requires the base dataset in memory.

```bash
python -m geoivf build --base data/base.fvecs --out data/geopack \
  --nlist 1024 --layout geopack --dims 16 --balls 8

# Strong contiguous-list baseline on this same physical layout:
python -m geoivf search --index data/geopack --queries data/query.fvecs \
  --out artifacts/baseline --backend native --direct --filter none \
  --nprobe 32 --window-pages 0 --max-extent-pages 256 --trace

# Same coarse candidate lists, with completion-driven page filtering:
python -m geoivf search --index data/geopack --queries data/query.fvecs \
  --out artifacts/filtered --backend native --direct --filter combined \
  --nprobe 32 --window-pages 64 --max-extent-pages 256 --trace
```

Use `--backend uring --direct --queue-depth 16` after the optional build.
`--gap-pages 1` permits reading one unnecessary page to join adjacent extents;
all such extra bytes and distances are counted. `--max-extent-pages` caps a
request's size. `--window-pages 0` means one whole-list planning window (subject
to the explicit seed reads used by filtered search).

New output/index directories must be empty. Results include `neighbors.npy`,
`queries.csv`, `run.json`, and optional `requests.jsonl`. Compare neighbor IDs
as well as recall against ground truth. The common kernel computes L2 in FP64
on unchanged FP32 payloads and breaks equal-distance ties by ID; Faiss can choose
a different ordering at floating-point ties. The Faiss integration test checks
untied fixtures; production comparisons must document their tie policy.

## Correctness and memory accounting

The first representation uses a variance-ranked subset of original coordinates.
This selection is a Euclidean contraction without estimating a PCA operator
norm. Each page has 1..capacity enclosing balls. Centers use shared-scale uint8
coordinates; radii use outward-rounded FP32. The radius is recomputed against
the original projected points AFTER center quantization. Query bounds are
computed in FP64 with a conservative numerical guard. This is an engineering
implementation of the covering invariant, not a formally verified floating-point
library. Boundary and containment tests accompany it.

The threshold starts at infinity and changes only after fetching and evaluating
actual vectors. No exact-neighbor oracle supplies the online threshold. Pages
are rejected only for `lower_bound > current_kth_distance`, retaining ties.
Candidate lists are fixed before filtering. Completion batches are processed in
canonical request order so backend completion order cannot change subsequent
read decisions. The current coordinator waits for a whole stage before planning
the next one; it is not a fully pipelined concurrent query engine.

For `b` balls and `m` retained coordinates, summary arrays cost `b*(m+4)` bytes per
page, not `b*(m+1)`: one-byte radii have NOT been implemented. The radial interval
adds 8 bytes/page. IDs, valid counts, list ranges, quantization scales and coarse
centroids also consume RAM. `manifest.json` separates summary bytes from the
complete NumPy directory size. Faiss routing makes an additional centroid copy;
Python/runtime and transient decoded arrays must be included in RSS measurements.
Do not call summary bytes the total index RAM budget.

## MQSim

```bash
bash scripts/setup_mqsim.sh
python scripts/mqsim_smoke.py --trace artifacts/smoke/geopack-combined.jsonl
```

This clones and builds upstream MQSim at commit
`51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16`. It preserves the upstream license and
records the revision with the output. The smoke configuration is deliberately
small and is NOT calibrated to the host's NVMe drive.

For a device-sensitivity experiment with an explicitly chosen arrival policy:

```bash
python -m geoivf export-mqsim --trace artifacts/filtered/requests.jsonl \
  --out artifacts/device-test --ssd-config third_party/MQSim/ssdconfig.xml \
  --request-gap-ns 100000
third_party/MQSim/MQSim -i third_party/MQSim/ssdconfig.xml \
  -w artifacts/device-test/workload.xml
```

The exporter writes MQSim's five fields: arrival time, device, starting LBA,
length in 512-byte sectors, and opcode (`1` means read). It derives resource IDs
from the supplied device configuration and rejects out-of-capacity addresses
rather than silently wrapping them.

**Critical limitation:** JSONL retains query IDs and read-stage dependencies,
but MQSim's ordinary trace format cannot express those dependencies. The exporter
uses explicitly synthetic fixed-spacing arrivals and marks the output
`open-loop-device-only`, `query_latency_valid=false`. This is useful for device
service/traffic sensitivity, not for reporting end-to-end ANN latency or QPS.
Changing SSD timings can change when a later stage may be issued. A future
closed-loop bridge or a real backend must enforce those completions. Never sum
MQSim I/O latencies to derive query latency, or relabel IOPS as ANN QPS.

## Execution discipline

The integration workflow runs only trusted pushes to `main` or manual main-branch
runs on the repository's `geoivf` runners. It does not run public pull requests on
self-hosted machines. It launches one integration job, limits thread counts and
build parallelism, and archives logs/results. No root operations or system cache
flushing are performed. Ten runners on one laptop are not ten independent SSDs:
use them for offline sweeps, not simultaneous latency benchmarks on one device.

Logical requested pages/extents are not necessarily physical NVMe commands.
Buffered reads may hit the OS cache; direct I/O still includes the kernel/device
stack and controller caching. Actual block-layer traffic and queueing must be
measured during the controlled real-NVMe evaluation. Keep trace collection separate
from final timing runs and report filter, routing, I/O, distance and total costs.

## Primary references

- Faiss IVFFlat and preassigned candidate semantics:
  https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFFlat.html
- Upstream MQSim, FAST 2018, trace format and device parameters:
  https://github.com/CMU-SAFARI/MQSim
- liburing read interface:
  https://man7.org/linux/man-pages/man3/io_uring_prep_read.3.html
- Direct-I/O alignment and filesystem limitations:
  https://man7.org/linux/man-pages/man2/open.2.html
