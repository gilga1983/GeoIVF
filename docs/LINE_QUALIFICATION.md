# Local-line representation and layout qualification

Implementation source: `e3c8e51253c5ead01bcec974afa882009d297748`.
This file describes the experiment, not an assertion that the pending run succeeded.

## Representation

For a contractive shared PCA transform, let z=P(x). Fit each page a free
least-squares affine line a+t*u, with ||u||=1. The mean is the origin and the
leading local principal direction is obtained from the small page Gram matrix.
Singleton and duplicate-point pages use a deterministic fallback direction.
These are free fitted anchors, not selected data-point endpoints or midpoints.

The deployed default stores an 8-bit origin per coordinate and a signed 8-bit
direction per coordinate. Directions decode through division by 127 followed
by normalization. Fit the per-point scalar descriptions AFTER decoding both
vectors: t=dot(z-a,u), e=z-a-t*u, rho=norm(e). Direction quantization is therefore
not silently ignored. No expanded origin/direction table is retained.

Store per point a uint16 t-bin and a uint16 rho-bin. The page additionally stores
three FP32 values: t-origin, t-step, rho-step. Rho bins start at zero. Quantizer
endpoints are outward-rounded and reconstructed scalar bins are expanded by
a numerical allowance. All original points are checked against the decoded
intervals during construction. Ranges use database points, not query data.

Default geometry bytes/page are 2*m + 4*capacity + 12, including scalar scales.
For eight-vector pages this is 172 B at m=64 and 300 B at m=128. Shared PCA,
global origin coordinate scales, valid counts, IDs, list ranges, IVF centers,
and the common 8-byte/page radial interval are additional and explicitly counted.
The FP32-axis diagnostic at m=64 costs 556 B/page, not 544 B.

## Two bounds from the same encoded line

For query projected coordinate tq and perpendicular norm rq, and enclosing
point intervals T=[tl,th] and R=[rl,rh], use

    lower_bound_shell = hypot(distance(tq,T), distance(rq,R)).

In exact Euclidean arithmetic this is safe because axial and orthogonal residual
components are orthogonal and ||eq-ei|| >= abs(||eq||-||ei||). Equivalently, the
map z -> (t(z),rho(z)) is nonexpansive. The scalar representation retains no
residual angular information. A line can be fitted in the full 128-dimensional
space without dimension reduction, but this angular information is still lost.

The matching ball ablation uses a center on the decoded line at (tl+th)/2 and
radius hypot((th-tl)/2,rh). It encloses the entire scalar-defined region but loses
the orthogonal structure. Its distance lower bound is

    max(0, hypot(tq-midpoint(T),rq) - hypot(width(T)/2,rh)).

Take the minimum over the active point slots and, in combined mode, the maximum
with the deterministic page radial bound. Reject only when this lower bound is
strictly greater than the threshold obtained from already fetched vectors.
The line shell bound is no weaker than this matching ball bound in exact
arithmetic. Unit tests check the implementation on ordinary and boundary inputs.

Query residual norms are evaluated from explicit residual vectors rather than
subtracting nearly equal squared norms. Floating-point guards cover numerical
error conservatively in the supported engineering setting, but this is NOT a
formally verified interval arithmetic implementation.

## Line-aware page layout

Initialize from the preceding immutable 16-original-coordinate GeoPack layout.
Run four alternating passes over adjacent full-page pairs within each IVF list.
For each pair of eight-vector pages:

1. Fit a free line to each current page in PCA64.
2. Score its sixteen vectors against both lines with an axial-overflow penalty.
3. Assign exactly eight vectors to each line by sorting assignment-cost differences.
4. Refit the two proposed pages and accept only a strict decrease in the sum of
   page objectives: sum(rho^2) + max(rho^2) + 0.1*(tmax-tmin)^2.

The proposal is a heuristic using fixed lines; the acceptance objective uses
refitted lines. This is a local, capacity-constrained refinement, not a globally
optimal partition, not exhaustive swaps, and not arbitrary selection from an
entire IVF list. Partial tail pages stay unchanged. Every list's exact multiset
of vector IDs is verified after packing. FP32 payloads are rewritten in the new
order without modifying coordinate values. Page count and list assignment stay
unchanged. Radial intervals are rebuilt for the new contents.

The legacy ball summary on the refined payload is deliberately disabled with
an infinite-radius fallback; a new line or independent-center sidecar is used
for every filtered comparison. This avoids retaining unsafe stale old bounds.

No queries fit the line models or layout. Minimizing line residuals is a design
surrogate; it does not guarantee better query pruning. The full-dimensional line
variant on refined pages still uses the layout optimized in PCA64, not a separate
128D-optimized layout. This distinction is part of the experiment's limits.

## Evaluation

Use the exact prior canonical SIFT1M base hash, nlist=1024, nprobe=64, k=10,
construction seed 12345, and the same 64 development queries. Fit shared global
PCA on the separate learning file, never the queries. All candidates are preassigned
and identical across methods. Compare both original and line-refined layouts.

Independent controls include PCA16/18/32 at eight bits; PCA64 at two/four/six/eight
bits; and PCA128 at two bits. These bracket the actual line storage costs. For
example, line64 costs 172 B and independent PCA18x8 costs 176 B. A budget of 176 B
admits both; they are not identical-size encodings. Line128 costs 300 B while the
PCA64x4, PCA32x8, and PCA128x2 controls each cost 288 B.

All filtered searches use 64-page windows, zero-gap coalescing, and the common
staged executor. The no-filter baseline permits whole-list reads. Compare
requested pages, bytes, extents, stages, distance computations, and memory.
Compare each returned answer with independent FP64 exhaustive evaluation over
its selected lists; independently audit rejection decisions on the first two
queries per configuration. Audits and configuration-query checks reuse queries
and pages; they must not be reported as independent samples.

Execution uses RAM payload replay. Interpreter timings are not native search
speedups. Neither physical NVMe latency nor CLIP performance is measured here.

## Reproduction

    python scripts/qualify_lines.py --work /path/to/empty-work \
        --out /path/to/empty-results --queries 64 --nprobe 64

The trusted-main workflow `.github/workflows/lines.yml` runs tests, validates the
shared dataset cache, executes the experiment, and uploads data provenance,
resolved packages, test reports, query IDs/routes, PCA, per-query CSVs, complete
aggregate JSON, and layout-refinement statistics. The model and layout tests can
also run without Faiss using the existing independent NumPy test routing path.
