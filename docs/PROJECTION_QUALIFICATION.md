# Conservative projection qualification

Implementation: `f21f0826138e7589088911577d7807cb8adad59c`.

This experiment follows the canonical qualification in which 16 selected original coordinates pruned very few pages. It separates loss of geometric information from quantization error and scheduling overhead. It is a count/correctness experiment, NOT a storage-latency benchmark.

## Unchanged components

The database is canonical TexMex SIFT1M. Its validated base-file SHA256 is checked, not merely its array shape. The full 1M FP32 payload is unchanged. Page capacity is eight 128D vectors in 4096 bytes. IVF has 1024 centroids, uses the preceding construction seed/sample, and all variants get identical preassigned candidate lists. The physical GeoPack layout is built with the original 16-coordinate partitioner and then frozen; it is NOT rebuilt in PCA space.

The query pool is the first 64 queries of the same deterministic 1000-query development split. It includes the 16 queries from the earlier qualification. The held-out 9000 queries remain untouched. PCA is fitted on the supplied 100000-vector learning file, not on development queries. Scalar quantizer ranges are derived from the indexed vectors only.

## Projection and safety

Let T be a full square transformation with operator norm at most one. Split its output into head coordinates H and remaining coordinates R. PCA uses covariance eigenvectors without whitening. The implementation bounds the norm of the stored FP64 matrix using the maximum absolute row sum of its Gram matrix, adds a floating-point allowance, and scales it down. It does not assume the numerically computed eigenvectors are exactly orthogonal.

For a group G on an IVF page, store a quantized head-space center a and an outward-rounded radius r covering the ORIGINAL transformed group points. Recompute the radius after quantization; do not cover already-quantized vectors and treat that as a cover of the original vectors.

The head lower bound is:

    h(q,G) = max(0, ||H(q-mu) - a|| - r).

For the optional tail representation, c is the IVF centroid of this page. Store an interval [l,u] covering all values ||R(x-c)|| for x in G. Its lower bound is:

    t(q,G) = distance(||R(q-c)||, [l,u]).

Then:

    L(q,G) = sqrt(h(q,G)^2 + t(q,G)^2)
    L(q,page) = min over groups G of L(q,G).

For every group member x, h is at most the head distance and t is at most the tail distance. Their squared sum is therefore at most ||T(q-x)||^2, which is at most ||q-x||^2. Combining is done PER GROUP before taking the minimum. A separate deterministic radial bound can be combined with the page bound using max, not a squared sum of overlapping information.

The online threshold is infinity until actual reads produce k candidates, then is the current kth fetched distance. Ground truth and final IVF radii are never inputs to online pruning. Strict > is the only rejection rule. Engineering FP64 error cushions and outward FP32 rounding are used; this is not a formally verified floating-point implementation. Tests include duplicates, zero vectors, partial pages, different center precisions, tail/no-tail, full-dimensional transforms, complete answer checks, and payload mutation checks.

## Representations and exact array costs

All variants here use eight groups per page. With singleton groups, the radius primarily records quantization/roundoff uncertainty; these are per-vector conservative proxies, not an eight-cluster compression of a much larger group.

| Representation | Geometry bytes/page |
|---|---:|
| Legacy original-coordinate 16D uint8 | 160 |
| PCA 16D uint8 | 160 |
| PCA 16D FP32 | 544 |
| PCA 32D uint8 | 288 |
| PCA 64D uint8 | 544 |
| PCA 8D uint8 + FP32 tail interval | 160 |
| PCA 16D uint8 + FP32 tail interval | 224 |

A FP32 radius costs four bytes/group. A tail interval costs eight additional bytes/group. The radial interval adds eight bytes/page to all combined variants. Full basis, mean, quantizer ranges, and transformed centroids are shared overhead and are accounted separately. IDs, occupancy, IVF centroids, and list ranges are included in directory_array_bytes. Runtime objects, query caches, routing duplicates, and build/oracle scratch are not represented by that array-byte count. The full projected database is build/diagnostic scratch ONLY, not a query-resident shadow index.

The PCA16 FP32 control asks whether uint8 error is the main limitation. PCA8+tail versus PCA16 is equal in geometric bytes, but not necessarily equal in shared metadata or CPU. PCA64 uint8 and PCA16 FP32 also have equal per-page geometry budgets.

## Scheduling controls

The existing search executor, readers, and distance code are unchanged. SeededNoFilter returns zero bounds but passes through the exact same seed-read and planning-window logic as filtered execution. This makes window boundaries and initial seed reads comparable. A whole-list no-filter baseline is retained as the more efficient contiguous baseline.

Windows of 16, 64, 256 and whole-list are examined. Gap bridging of one and two pages is examined separately. All gap pages, bytes, and distance evaluations are counted. Scheduling probes use the first 16 development queries; the main representation comparison uses 64. Do not mix those sample sizes when comparing aggregates.

## Offline diagnostic

`offline-headroom.json` uses the FINAL IVF kth distance and exact, unquantized PCA coordinates to estimate how many pages an ideal head-only proxy could reject at dimensions 16, 32, 64 and 128. This is optimistic diagnostic information, not an online algorithm, not a measured speedup, and not a bound on methods retaining additional geometry such as tail intervals.

## Reproduction

    python scripts/fetch_sift1m.py --report artifacts/dataset.json
    python scripts/qualify_projections.py --work /path/to/empty-work \
        --out /path/to/empty-output --queries 64 --nprobe 16 64

`Projection qualification` runs trusted main-branch pushes/manual dispatches on the geoivf runners. It builds pinned liburing, runs the complete test suite, validates the shared dataset cache, runs the experiment, and archives results. Only its uniquely created temporary work directory is cleaned up. The cached canonical dataset is never deleted.

The result bundle contains source commit, package versions, JUnit tests, dataset manifest, exact query IDs/routes, PCA basis, per-query CSVs, aggregate JSON, offline diagnostics and the console log. Actual I/O and CLIP performance comparisons remain separate gates.
