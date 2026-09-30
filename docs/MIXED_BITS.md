# Mixed-bit scalar coding on fixed IVF pages

This is a controlled representation experiment, not an implementation or novelty
claim for SAQ. The existing PCA64/SQ8 and full-dimensional PQ variants remain
explicit baselines. No previous default or search contract is changed.

## Encodings and budget

Each vector occupies exactly 512 coordinate-code bits (64 bytes), followed by
its existing outward-rounded FP32 reconstruction-error radius. At eight vectors
per page this is 544 geometry bytes, excluding common radial intervals, vector
IDs, coarse centroids and shared model metadata. Allocation is global per index,
not per vector/page, so the record size and address arithmetic remain fixed.

Uniform controls: PCA64/SQ8, PCA128/SQ4, original128/SQ4 and plain PQ64x8.
Every mixed variant retains ALL 128 PCA coordinates with widths in 2/4/6/8:

- Manual: first32 coordinates use8 bits, next32 use4, final64 use2.
- Variance: exact discrete allocation minimizing sum_j var_j * 2^(-2*b_j).
- Measured-MSE: exact discrete allocation minimizing measured coordinate error
  under the DEPLOYED clipped uniform scalar quantizers.
- Grouped measured-MSE: same objective, constrained to at most four contiguous
  segments with boundaries at multiples of eight coordinates.

The discrete optimizer exhaustively solves its additive distortion objective
under the stated width, budget and segmentation constraints. It does NOT solve
an optimal recall, tail-radius, verification-I/O or latency objective. Small
instances are checked against complete enumeration. Uniform4 is a feasible
control for both MSE optimizers. All encodings count the same 512 bits; selecting
zero-bit/discarded coordinates is deliberately excluded from this experiment.

## Calibration and error certification

PCA and bit-allocation distortion/variance statistics use the separate SIFT1M
100000-vector learning file. Scalar ranges use minima/maxima from the indexed
base corpus, preserving the existing uniform-quantizer calibration policy.
No query trains PCA, allocation or scalar bins. PQ uses the same50000 learning
vectors, seed and final20-iteration codebook training as the preceding study.

Pack codes into actual fixed-length records, decode them, and recompute each
point's reconstruction-error radius. Audit every indexed point after decoding.
Low-bit coordinates contribute their full errors to the radius; retaining a
coordinate with two bits does not remove its uncertainty. Full-space transform
norm bounds and query/build guards use the existing rotation verifier.

## Execution and controls

The new scalar native scanner specializes contiguous segments for 2/4/6/8-bit
unpacking. It computes lookup tables inside the measured query, accumulates in
FP64 coordinate order and retains the same shortlist/upper-bound heap logic as
the preceding uniform scanner. It does not keep expanded per-vector centroids.
All uniform SCALAR controls use this same grouped scanner. PQ retains its existing
uniform-code LUT scanner, using64 group contributions rather than128 scalar
contributions. The ranker here is not a Faiss FastScan benchmark.

The previous PCA64 lookup scanner is rerun in separate controls. Tests and the
canonical campaign compare full staged request-plan hashes on identical codes,
so a different native decoder cannot be credited as better geometry. Mixed
allocations may have many short segments; their real cost is measured rather
than assuming unequal precision is free. Lookup tables are query scratch;
width/segment descriptors are charged as shared arrays.

All modes call the UNCHANGED search_rotation page-verification coordinator,
SIMD original-FP32/FP64 exact scanner and pooled rolling io_uring/O_DIRECT reader
at queue depth16. Approximate mode verifies the shortlisted pages in one wave;
certified mode reads all unresolved pages after the first exact threshold.
No gap bridging. Queries route afresh inside timing; no oracle threshold is used.

## Canonical experiment

Freeze original SIFT1M payload, page grouping/order, nlist1024, nprobe64, k10 and
seed12345. Development queries are permutation(seed20260929)[:128]. Tune R over
16/32/64/128/256 separately by representation/mode to meet >=99% strict recall,
then freeze choices before evaluating fresh queries permutation[1768:2024].
This excludes every previously used development/test cohort. Do not retune on
these test queries and later call them untouched.

The82 development arms include80 family/mode/R combinations plus two old-scanner
controls. Held-out arms include frozen approximate/certified settings, all
predeclared approximate R64 controls and the old-scanner controls. Three randomly
interleaved rounds measure direct-I/O and RAM replay separately. All tested
certified answers must match the independent exhaustive selected-IVF-list oracle.
Approximate misses are saved with neighbor IDs, not hidden by aggregate recall.

Report geometry/radius distributions, actual allocations and segment counts,
learning distortion, selected and fixed-R recall, pages/extents/decision waves,
ranking time and full-query latency. Compare same-campaign controls only.
The machine is shared; affinity and a GeoIVF file lock do not isolate it from
other work. Database/oracle/variant arrays in benchmark RSS are not deployment
memory. No larger-than-RAM, matched total RSS, concurrent throughput, physical
NVMe-command or fresh external SOTA comparison is established by this campaign.

Build with `bash scripts/build_mixedbits.sh --uring`. Run the campaign with
`scripts/qualify_mixedbits.py --work NEW_PRIVATE_DIR --out NEW_RESULTS_DIR`.
Use `MixedIndex(layout, sidecar)` with the existing `search_rotation` API for
standalone mixed-bit searches. Tests also exercise upper-bound verification,
but the canonical performance campaign focuses on approximate/two-wave modes.
Results and their artifact checks are documented separately after completion.
