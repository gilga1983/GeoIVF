# Dependent-center qualification

The experiment keeps the canonical SIFT1M payload, original frozen GeoPack page
membership, shared PCA, IVF candidates and filtered-search schedule unchanged.
It tests center compression, not a new page layout or an SSD speedup.

Implementation commit: `241539bc59fe6a0aa3f6bceb023c2b45435c0f5f`.

## Representations

Each 4 KiB page contains up to eight original 128D FP32 vectors. This experiment
uses one projected ball per vector. Explicit anchors are stored as shared-scale
8-bit coordinates. Other centers may be derived as the midpoint of two anchors,
or as an interpolation with t in {0, 1/16, ..., 1}. Pair identifiers occupy one
byte per derived center; interpolation coefficients use another byte. These
identifiers are stored, not treated as free metadata. No vector-to-anchor mapping
is needed by the page-union bound: the ball list can be reordered independently
of payload slots, which remain unchanged.

The two residual variants encode an additional correction per derived center.
Two-bit codes use three signed levels {-1,0,1}, with one unused code; four-bit
codes use fifteen signed levels {-7,...,7}, again with one unused code. Each
residual vector has a four-byte scale. All residual coordinates are truly bit
packed. Baseline independent codes are packed at 4, 6 or 8 bits, with global
per-coordinate min/max scales. Neither 4-bit residuals nor independent 4-bit
coordinates are stored as a byte per coordinate.

All radii are four-byte outward-rounded floats. Shared projection/scales and the
common eight-byte radial interval are reported separately from geometry bytes.
Full allocated directory arrays are also reported. Runtime/object overhead and
decoded query-window scratch are additional. Centers are NOT permanently
expanded to FP64 at index load.

## Construction and guarantees

For A anchors chosen from the page's original projected points, all size-A
subsets are enumerated. For each subset, every remaining point chooses the
allowed pair/weight minimizing its squared prediction error. The subset with
minimum total squared center error is selected. This is an exhaustive optimum
for that finite data-point-anchor and fixed-weight-grid surrogate in ordinary
floating-point arithmetic, NOT an optimum for query false positives or arbitrary
free anchor locations. Short partially populated pages receive explicit handling.
Residual variants reuse the interpolation-selected anchors; they do not jointly
optimize the residual coding objective.

Radii are computed against the original projected vectors after decoding anchors,
interpolation and residual corrections exactly as the query decoder does. The
builder independently checks coverage of all one million indexed points by the
decoded ball union. Query rejection uses the existing conservative bound and
strict lower_bound > current threshold condition. The online threshold comes
only from actually fetched vectors; no final-radius oracle supplies it.

PCA fitting, contraction normalization and engineering numerical guards are
reused from the preceding projection experiment. These measures are not a
formally verified floating-point arithmetic library.

## Canonical evaluation

Run with:

```bash
python scripts/qualify_dependencies.py \
  --work /path/to/empty/workdir --out /path/to/empty/results \
  --queries 64 --nprobe 64
```

The dataset cache defaults to ~/.cache/geoivf/datasets/sift1m. The experiment
uses nlist=1024, k=10, seed=12345, and the same 64 development queries as the
preceding PCA experiment. It asserts the original payload hash. Every result is
compared with independent exhaustive FP64 evaluation over identical preassigned
IVF lists, and rejected pages for the first two queries are independently audited.
Only the selected 64 query vectors are used for evaluation, never for fitting.
No held-out generalization claim is implied.

The output includes encoding byte breakdowns, error/radius statistics, all
per-query counts and timings, provenance, PCA matrix, query IDs and routes.
Timings from the Python/NumPy memory-replay harness are diagnostics, not native
SSD latency or throughput. The no-filter control receives whole-list planning;
all filtered encodings receive the same 64-page window and zero-gap coalescing.

The unit suite includes packed-code round trips for 1 through 8 bits, exact
collinear midpoint/interpolation examples, partial pages, cover/bound validity,
unchanged payload hashes, byte accounting and search-result equivalence.

## Scope of interpretation

Poor midpoint results would reject this particular within-page data-anchor
construction, not all possible shared codebooks, fitted free anchors, geometry-
aware page redesign, or compression of multi-point balls. Likewise a residual
variant must beat independent lower-bit quantization before its dependency
complexity is justified. Always include that control in result tables.
