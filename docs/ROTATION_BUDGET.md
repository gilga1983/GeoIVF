# Equal-budget rotations and full-space verification

This is an opt-in representation experiment. It does not change the SSD FP32
payload, GeoPack page membership, IVF centroids/assignments, or existing defaults.

## Representations and accounted bytes

All seven variants store exactly 64 coordinate/code bytes per vector and one
outward-rounded FP32 reconstruction-error radius. Eight slots/page cost 544
geometry bytes, excluding the common radial interval and structural/shared data.

- PCA64 + uniform 8-bit scalar quantization: existing subspace/code reference.
- PCA64 + a seeded random orthogonal rotation within that subspace + SQ8.
- Centered original 128 coordinates + uniform SQ4.
- Full PCA128 + uniform SQ4.
- Seeded full 128D random orthogonal rotation + uniform SQ4.
- Faiss PQ64x8, without a learned rotation; 64 two-coordinate codebooks.
- Faiss OPQ64 followed by PQ64x8, retaining all 128 dimensions.

PCA is fitted on the separate 100000-vector SIFT learning file. PQ and OPQ use
the same fixed 50000-row learning subset. The pilot fixes 20 OPQ outer iterations,
20 initial and four subsequent PQ iterations, followed by a 20-iteration final
PQ fit. This is not a claim of exhaustive OPQ tuning or use of its default
training schedule. Both PQ and OPQ final codebooks use the same training budget.
Uniform scalar ranges are calibrated from the base corpus, not test queries.
Model means, matrices, codebooks and scalar range arrays are counted separately;
no permanent expanded per-vector reconstructions are retained at query time.

The PQ/OPQ training and assignment use Faiss; search uses our common FP64 table
accumulator for both scalar and product codes. Tables are built afresh inside
every query timing. This is not a claim to benchmark optimized Faiss PQFastScan.
The new table scanner is an EXECUTION change distinct from rotation. The prior
staged/approximate/certified PCA64 paths are rerun as controls. The new PCA64
packed coordinate bytes must exactly match the previous sidecar.

## Numerical and geometric contract

For the deployed row transform T, z=(x-mean)T, let beta bound ||T||_2 above.
For full square transforms only, let alpha>0 bound the least singular value
below. Conservative Gram-matrix row sums/Gershgorin bounds plus floating-point
allowances establish beta and alpha. A rectangular projection has alpha=0,
not the smallest positive singular value of its output-space Gram matrix.
Do NOT use projected upper bounds as full-space upper bounds.

Store r >= ||z-decoded_code||, including the deployed transform, quantizer,
assignment rounding and outward radius rounding. For a query y=(q-mean)T,
a decoded-center distance d produces conservative bounds

  L = max(0,(d-r-query_numeric_guard)/beta)
  U = (d+r+query_numeric_guard)/alpha, when alpha>0.

Page lower bounds are the minimum of its per-point L values, combined with
the existing deterministic centroid-radial page interval. Guarded computations
are engineering safeguards, not a formally verified floating-point library.
Every built scalar/PQ encoding is decoded from its actual packed records and
checked for coverage of every indexed vector before search.

## Three verification contracts

Approximate: retain R compressed-score candidates, fetch their unique pages,
and exactly rank all original vectors on those pages. Missing candidates may
reduce recall. No inverse rotation is needed or performed.

Certified: retain the same shortlist, fetch it, then compute tau from verified
full-vector distances. Fetch every unread candidate page with L<=tau as a second
fixed wave. This preserves the full-vector answer within the selected IVF lists.

Upper-bound one-wave: full-dimensional representations compute the kth smallest
U among DISTINCT candidate vectors. At least k true vectors have distances no
greater than this threshold. Fetch every page with L<=that threshold. Exact
ranking of the fetched FP32 vectors is therefore complete within the selected
IVF lists. An upper bound that is loose can force MANY page reads. A single wave
is not one SSD command and can be split into bounded reader batches. The threshold
is not retuned after each batch, so this remains one data-dependent decision wave.

All verification uses the existing SIMD FP64 scanner and pooled rolling
io_uring/O_DIRECT reader. No oracle contributes any search threshold. Tail pages,
unique reads, strict rejection, and ID tie-breaking remain explicit invariants.

## Experiment protocol

Full canonical SIFT1M; unchanged nlist=1024, nprobe=64, k=10 and construction seed.
Fixed page payload SHA256 must match preceding campaigns. Development uses 128
queries from permutation(seed=20260929)[:128]. Sweep R=16/32/64/128/256 for each
representation and approximate/certified mode. Run the full-dimensional one-wave
option separately. Three randomized interleaved rounds select the fastest setting
in each representation/mode family meeting the predeclared 99% strict Recall@10
target. Infeasible targets are recorded, not silently treated as achieved.

The fresh held-out cohort is permutation[1512:1768]. It excludes the original
1000-query development pool and BOTH previous 256-query test cohorts. Freeze
shortlist choices before evaluating these queries. Also predeclare an R64
approximate arm for every representation to isolate code quality at fixed R,
and retain all applicable one-wave upper-bound arms. No held-out retuning.

The direct-I/O and memory-replay phases both include fresh routing, projection,
lookup-table construction, ranking, verification and exact scan. Checks are
outside timing. Record per-query neighbor IDs, recall, pages/bytes/extents/waves,
CPU and wall time, build time, code/transform hashes and memory components.
Audit rejected pages against original vectors only after search, on the first
two development queries per exact arm. All exact-mode answers and repeated
operation counts are checked on every measured query.

A host lock serializes GeoIVF speed campaigns, not unrelated processes. One CPU
is pinned but not reserved. The payload fits RAM; original vectors and the oracle
coexist in benchmark memory. O_DIRECT bypasses the host page cache, not device
caches. No equal-RSS, larger-than-RAM, concurrency, physical-command or new
external-SOTA claim follows from this pilot. Compare methods within this run.

Build: bash scripts/build_rotations.sh --uring
Execute: python scripts/qualify_rotations.py --work NEW_WORK --out NEW_RESULTS
The workflow .github/workflows/rotations.yml runs this on existing GeoIVF runners.
