# Canonical-data qualification

This milestone separates immutable SSD page layout from RAM representation.
The legacy `geoivf build` command is unchanged; it is not used for independent
layout/summary ablations. Use `freeze_layout` and `summarize` from
`geoivf.layouts`, or the canonical qualification driver below.

## Dataset and execution

```bash
python scripts/fetch_sift1m.py --report dataset-manifest.json
python scripts/qualify_sift1m.py --work /path/to/new/scratch \
  --out artifacts/sift1m-dev --queries 32 --nprobe 4 8 16 32 64 128 256
```

The source is the canonical TexMex ANN_SIFT1M archive, not a SIFT1B subset or
newly extracted image descriptors. The downloader validates the dimensions,
record counts, file lengths, finite coordinates, and ground-truth ID range.
It records SHA256 values for the archive and files. These are observed integrity
hashes, not signatures or comparison with published upstream checksums.
A process lock prevents concurrent downloads into the shared cache. Extraction
copies only the four explicitly expected regular members, never archive paths.

The initial Actions qualification deliberately uses 16 development queries and
nprobe 4/16/64. This is a first canonical-data gate, not the complete campaign.
The driver supports the full sweep above, additional queries and construction
seeds. The split is a fixed seed-20260929 permutation: first 1,000 queries for
development, remaining 9,000 for held-out testing. Use `--split heldout` only
when the configuration is frozen. Do not tune parameters on those results.

The coarse index has 1,024 lists by default. Faiss trains from a fixed 100,000
vector sample of the base data. Every layout shares the same centroids,
assignments, query IDs and preassigned candidate lists. No test query influences
page construction, coordinate selection, or quantization scales.

## Ablations

The fixed-representation matrix is input/radial/GeoPack layout crossed with
none/radial/balls/combined filtering, using 8 balls and 16 selected coordinates.
This is 12 configurations for every nprobe. GeoPack is built once with 16 packing
coordinates. The additional RAM sweep re-summarizes exactly the same GeoPack
pages with 4/8/16 coordinates. All variants retain unchanged FP32 payloads.
Hash checks and page-ID equality tests enforce the fixed-layout comparison.

The radial filter is deterministic reverse-triangle pruning. It is not CLIP.
The script separately builds pinned upstream CLIP and HIVF-CLIP binaries; that
build alone does not constitute a CLIP benchmark or I/O integration.

Whole-list planning and coalesced reads are available to the no-filter baseline.
Filtered search currently uses 64-page windows in the initial qualification.
For final timing, explicitly sweep planning windows and gap bridging using the
search command. Baseline and filtered execution must have access to the same
I/O scheduler and request-size limits.

## Correctness and metrics

All answers are checked against independent FP64 distance evaluation of the
same selected IVF lists, with deterministic ID tie-breaking. Absolute recall
is separately measured against the published top-10 ID ground truth. Two queries
also undergo a full-database distance check of the ground truth, permitting
alternative IDs only at exactly equal distances in this diagnostic.

For the first two development queries per configuration, every rejected page is
inspected after search. Its true minimum distance must exceed the threshold at
rejection, and its computed bound must not exceed the true minimum distance.
These audit reads do not feed the online threshold and are not counted as search
I/O. Audit recording adds overhead, so the resulting times are diagnostic only.

Per-query CSVs distinguish selected pages, requested pages after coalescing,
read extents, bytes, stages, and distance evaluations. The initial driver uses
MemoryReplay to qualify the algorithm. It does NOT measure NVMe latency or
physical device commands. Flags `storage_latency_valid=false` and
`production_speedup_valid=false` are deliberately retained in its report.
Preassigned routing computation is outside the measured search loop.

Summary metadata is b*(dims+4) bytes per page: uint8 coordinates plus FP32 radius.
The radial interval is another 8 bytes per page. Allocated directory-array bytes
include common IDs/counts/ranges/centroids; Python overhead and decoded temporary
arrays are additional. The benchmark process's maximum RSS includes the base
vectors, correctness oracle and RAM payload replay. It is not deployed index RAM.
All summaries are loaded even in no-filter runs: allocated RAM and a hypothetical
minimal baseline RAM footprint must not be confused in a future comparison.

## Native asynchronous validation

```bash
bash scripts/setup_uring.sh
GEOIVF_TEST_DIRECT=1 GEOIVF_TEST_URING=1 python -m pytest -q
```

Upstream liburing is pinned at commit
`08468cc3830185c75f9e7edefd88aa01e5c2f8ab`. It is built in the checkout with PIC;
no sudo, system install, or host configuration change is required. Tests use
unique page contents, mixed extent lengths, multiple queue depths, EOF/error
propagation, poisoned-reader rejection, and end-to-end search equivalence.
The adapter submits bounded batches then waits for the batch. It does not yet
refill a rolling queue on each completion or pipeline multiple queries.

## Upstream CLIP

```bash
bash scripts/setup_clip.sh
```

CLIP is pinned at `7f4fc84edffede0aa21fae6131ec7391ce99ab6f`, including its
recorded Faiss submodule. Its source/build configuration is not silently rewritten.
System BLAS/LAPACK development libraries may be required. Failures are archived,
not reported as a successful integration. Search semantics and paper-level
performance comparisons require subsequent canonical-data execution.

## Resource discipline

Only trusted main-branch pushes/manual runs use the self-hosted runners. The
canonical job and a capped two-compiler upstream-build job may overlap, so none
of their timings is a controlled performance result. Do not run independent
NVMe latency experiments concurrently on these runner installations.
Results/packages/commit information are archived; scratch payloads live under
RUNNER_TEMP and canonical downloads in the shared user cache.
