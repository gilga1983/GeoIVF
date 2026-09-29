# Implicit quantization-cell qualification

## Representation contract

Keep the existing full FP32 payloads, frozen GeoPack membership, IVF centroids,
preassigned lists, PCA basis, and independently packed scalar codes unchanged.
The current nearest-rounding encoder uses a shared origin o_j and step s_j.
A decoded center coordinate c_j = o_j + code_j*s_j implies the closed interval
[c_j-s_j/2, c_j+s_j/2] provided no out-of-range input was silently clipped.
A small engineering roundoff pad enlarges each interval. The builder explicitly
checks every indexed projected coordinate against its decoded interval and
refuses a failing construction. The certificate binds the exact sidecar file
and page payload by SHA256. This is a static-index contract, not an insert or
out-of-distribution overflow policy.

One code denotes one point's Cartesian-product cell, not a page-wide bounding
box. Invalid tail-page slots are excluded using existing valid counts.
No explicit endpoint arrays are stored. Pure box mode does not retain a radius
array in its RAM sidecar. Radius data remains available in the shared experiment
input file for the alternative ball/hybrid loads; the claimed reduction is RAM,
not deletion of that shared research input file.

For y = Pq, define g_j = max(0, abs(y_j-c_j)-s_j/2-pad).
The box lower bound is sqrt(sum_j g_j^2). Since P is a contraction and the
projected indexed point is in the cell, this is conservative for original L2.
The existing measured-error ball gives max(0, ||y-c||-r).
Hybrid takes their maximum PER POINT, then the minimum over the page. This is
safe and no weaker than either constituent bound; it is not the exact distance
to the cell/ball intersection. The common deterministic page-radial bound can
also be combined. It is NOT CLIP.

These are engineering FP64 safeguards, not formally verified interval arithmetic.
The compiler is instructed not to enable fast-math or expression contraction.

## Memory

For eight singleton proxies, 64 coordinates and b bits/coordinate, packed
coordinates use 64*b bytes/page. Balls add eight FP32 radii, or 32 bytes/page.
Box geometry therefore costs 256/384/512 bytes at 4/6/8 bits; ball and hybrid
geometry cost 288/416/544. The common radial interval adds eight bytes/page.
The separately reported total directory also includes IDs, counts, list ranges,
IVF centroids, shared PCA and quantizer arrays. Runtime objects and transient
buffers are extra. No-filter runs in this diagnostic harness still load a
legacy directory and must not be presented as a minimal-memory baseline.

## Two distinct experiments

Online correctness and page accounting use the existing staged search. Its
threshold starts at infinity and changes only after genuinely fetching and
examining full payload vectors. Native full-bound kernels implement each
representation through the common executor; the exact scan/top-k path remains
Python/NumPy. Independent FP64 exhaustive candidate-set search verifies IDs,
with ID tie breaking. Rejection audits read original vectors after the search.
The legacy NumPy ball control must reproduce the native ball's non-timing counts.

The CPU experiment is a separate mechanism microbenchmark on the first 32 of
the same development queries. All shapes receive the identical candidate-page
arrays and final candidate-set radii, and use seven randomly interleaved rounds.
Those final radii are OFFLINE diagnostic inputs, never fed to online search.
The reported times include the Python/ctypes call, query projection, validation,
allocation, and mask checksum; they exclude routing, radial filtering, payload
I/O and exact top-k. Data/library pages are warmed, not guaranteed to fit in cache.
One shared host is measured; no exclusive core, fixed-frequency or thermal
control is asserted. Timings are not ANN latency or SSD speedups.

Both native selection strategies avoid square roots for ALL shapes:
- Ball: sum(delta_j^2) > (tau+guard+r)^2 rejects.
- Box: sum(g_j^2) > (tau+guard)^2 rejects.
- Hybrid: either inequality rejects a point.

The fixed-work strategy evaluates all dimensions of ALL active centers, even
when one center has already admitted the page. This isolates per-code arithmetic
more cleanly. The adaptive strategy checks every 16 dimensions, can reject a
center early, and can admit a page after one completed center survives. It may
perform different amounts of work for different shapes. Equal decision masks
are verified against full-bound evaluation. Generic -O3 native loops are not
hand-written SIMD or lookup-table implementations, so this experiment cannot
establish that either geometric primitive is universally faster on all CPUs.

## Reproduction

```bash
bash scripts/build_cells.sh
make -j2
GEOIVF_TEST_DIRECT=1 python -m pytest -q
python scripts/fetch_sift1m.py
python scripts/qualify_cells.py --work /tmp/geoivf-cell-work \
  --out artifacts/cells/results --queries 64 --cpu-repeats 7
```

Use new empty work/output directories. The workflow runs only trusted pushes to
experiment/quantization-cells or explicit permitted manual invocation. It is
single-job and does not launch benchmark matrices across runners sharing a disk.
All variants use one million canonical SIFT1M vectors, one construction seed,
1024 IVF lists, nprobe=64, k=10, and the same reused 64 development queries.
The held-out 9000-query partition is not consulted. Results need broader held-out
and multi-dataset validation before publication claims.

Prior-art context: implicit rectangular approximation cells with safe distance
bounds are a VA-file-style primitive. The experiment tests that primitive in our
fixed-page PCA directory and execution path, not a claim that cell filtering is
new. GCC floating-point flag definitions: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
