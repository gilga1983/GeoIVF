# Compact short-segment packing results, 2026-09-29

## Provenance and scope

Tested implementation: `9f721226db5040e13ed3efd03e7be2a2970ccfda`.
Workflow: `36535930605`, Compact-segment qualification.
Artifact: `11018383238`, `compact-segments-36535930605`.
Archive SHA256: `45fed91e1554204c2f086df790adce231091861feacb36d450d7e10c49eff578`.
The workflow completed successfully, including native direct I/O and io_uring
regressions. The canonical qualification itself uses RAM payload replay, NOT
measured SSD traffic, latency, or throughput. No CLIP or held-out comparison.

Canonical TexMex SIFT1M, all 1,000,000 unchanged 128D FP32 vectors, 4 KiB pages,
eight vectors/page, 125445 pages, nlist=1024, nprobe=64, k=10, seed=12345.
The same 64 development queries, learning-file PCA and preassigned candidate
lists as the preceding experiments were retained. Every configuration has
Recall@10=0.99375 and the same independent exhaustive IVF-candidate-set answer.

Source GeoPack payload: `8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.
PCA basis: `d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64`.
Candidate routes: `16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b`.
All match the preceding line study. All ten repeated original/old-linepool
settings exactly reproduce their prior non-timing page/extent/stage/distance counts.

## What changed

Instead of choosing page companions by perpendicular line error first, generate
balanced 8/8 proposals from short axial windows, joint axial/residual cost,
full-space center proximity, and nearest-to-seed subsets. Exchanges occur only
inside 64-vector within-IVF-list pools. Seven matching rounds visit all page
pairs in each pool once. The best four feasible proposals are freely refitted.

Strict variant: every page slot must have full 128D diameter no larger than its
ORIGINAL GeoPack diameter. Relaxed variant permits at most 5% growth against that
same fixed original budget. Budgets do not compound across updates. Reject
infeasible candidates BEFORE ranking them. Accept feasible changes only when
the paired fitted objective `L^2 + 4*r_max^2` in PCA64 decreases.

All IVF memberships, vector multiplicities, page counts, capacities and partial
tail pages are preserved. The prior builder used a different residual-heavy
objective and no hard diameter cap. These are different local heuristics, not
an isolated sweep of one objective coefficient or globally optimal packing.
Line axes remain least-squares fits, not minimax-optimal short-segment fits.
Default GeoPack behavior is unchanged; all new construction is opt-in.

## Geometry really becomes more compact

Mean values across all pages; diameter is measured in original 128D space, while
segment span and maximum perpendicular residual are measured in PCA64.

| Layout | Full-space diameter | Segment length | Maximum perpendicular residual |
|---|---:|---:|---:|
| Original GeoPack | 354.3841 | 282.9754 | 181.0593 |
| Previous line-first pool64 | 345.5173 | 281.4983 | 167.0403 |
| Compact, strict diameter | 335.7397 | 252.4986 | 169.0948 |
| Compact, 5% allowance | 336.9781 | 252.7440 | 168.1486 |

The strict variant reduces mean diameter 5.261%, mean segment length 10.770%,
and mean maximum residual 6.608% relative to original GeoPack. Its tube objective
falls 15.271%. It changes membership on 115681 pages (92.217%). Every page obeys
the strict diameter constraint: maximum observed new/original ratio is 1.0.

The 5% variant changes 117861 pages (93.954%); the maximum diameter ratio is
1.049994537. It improves the mean tube objective 15.792%, but does not shorten
segments further than the strict variant in this run. These invariants were
checked during construction and again from the downloaded per-page arrays.

The previous line-first builder also reduced average diameter, by 2.502%,
although its diameter p95 increased from 431.5180 to 441.5366. Therefore a claim
that its disappointing filtering was simply caused by universally longer or
less compact pages would be unsupported.

## Actual requested pages at fixed answers

Every unfiltered layout requests 8394.796875 pages/query. Each filtered method
uses the same 64-page windows and zero-gap coalescing. Geometry includes actual
per-page encoding/quantizer parameters, but excludes the common 8 B/page radial
interval, shared PCA, and structural metadata. No-filter metadata allocation in
this harness is not a minimal-RAM implementation.

| Layout | Line64 shell, 172 B: pages/query | Page reduction | Median per-query reduction | Independent PCA18/8-bit, 176 B: pages/query | Page reduction |
|---|---:|---:|---:|---:|---:|
| Original GeoPack | 7754.8438 | 7.6232% | 0.5245% | 5732.0625 | 31.7189% |
| Previous line-first pool64 | 7418.5625 | 11.6290% | 2.5854% | 5646.0781 | 32.7431% |
| Compact, strict diameter | 7532.5938 | 10.2707% | 1.3306% | 5660.0625 | 32.5765% |
| Compact, 5% allowance | 7514.6719 | 10.4842% | 1.3940% | 5660.2188 | 32.5747% |

Independent PCA64 at four bits (288 B/page) reads 2189.4844, 2102.2813,
2140.0156, and 2137.2188 pages/query on these four layouts, respectively.

The compact variants improve the line over original GeoPack but do not beat the
previous line-first heuristic. They remain substantially less selective than
near-budget independent centers. They also give small improvements to the
independent representations relative to original GeoPack.

For the strict variant, requested extents/query are 419.6094 for Line64,
741.2188 for PCA18, and 878.1563 for PCA64/four-bit. The unfiltered whole-list
baseline uses 65.0938. Thus this is NOT a dominance claim on every metric or
an end-to-end speed ranking. More selective independent codes can cause more
fragmented request sets.

## Separating neighbor grouping from false-positive filtering

Offline only: hold the query radius fixed to the exhaustive candidate-set kth
radius tau, then count pages containing at least one actual in-radius vector.
Also examine 1.1*tau and 1.25*tau. These thresholds never guide online search.

| Layout | Actual-neighbor pages at tau | At 1.1*tau | At 1.25*tau |
|---|---:|---:|---:|
| Original GeoPack | 9.265625 | 84.531250 | 950.187500 |
| Previous line-first pool64 | 8.890625 | 80.312500 | 896.296875 |
| Compact, strict diameter | 9.125000 | 83.125000 | 926.140625 |
| Compact, 5% allowance | 9.250000 | 82.625000 | 924.203125 |

The compact grouping mildly improves these counts versus original GeoPack, but
the previous line-first layout concentrates these query neighborhoods better
despite having a larger mean diameter. Smaller diameter is a useful proxy, not
a guarantee of better co-access locality. This is one reused development sample,
not a general ordering of all possible geometries or workloads.

For strict compact pages with the line shell, mean requested pages decompose as:

`7532.593750 = 9.125000 true-neighbor pages
            + 7521.171875 false-positive pages at the final radius
            + 2.296875 extra reads above that final-radius admission count`.

The last term includes actual seed reads and looser online thresholds under the
chosen execution order. It is not an SSD scheduling-latency estimate. This shows
that almost all remaining line traffic is still admitted by the summary even
at the final threshold, rather than being caused by slow threshold warm-up.
Near-budget independent PCA18 on exactly these pages admits 5636.15625 false-
positive pages at the final radius; its online total is 5660.0625.

The strict page line explains mean 30.055% of local variation, versus 30.983%
for original GeoPack and 36.843% for the previous line-first layout. Mean decoded
perpendicular distances are 137.391, 140.151, and 130.783, respectively. Shorter
segments have not made the residual cloud thin enough. Loss of residual angular
information remains consistent with the diagnostics, not a proven unique cause.

## Correctness and verification

105 runner tests passed with zero failures and zero skips, including the nine
new packing tests and the previously validated O_DIRECT/io_uring paths.
All 1280 configuration-query comparisons (20 settings on 64 distinct queries)
matched the independent FP64 candidate-set oracle with ID tie-breaking.
88731 rejected-page decisions were audited against original vectors after search;
all passed. These audits cover the first two queries in each configuration and
include repeated pages; they are not independent query/page counts.

The existing line builders checked scalar-interval coverage of every indexed
point in every built line encoding. Independent-center builders checked their
all-point covering invariant. Offline final-radius tests retained every actual
in-radius page for all 64 queries/configurations. Numeric guards are engineering
safeguards, not a formally verified floating-point library.

After download, ZIP SHA256, all CSV query identities/means, geometry byte sums,
byte/page equations, the three-term page decomposition, per-page diameter caps,
and old-baseline non-timing counts were independently reconciled. The artifact
also contains the complete tested source archive. Temporary payload copies were
removed after upload; the shared canonical dataset cache was retained.

## Decision

The experiment implements the requested locality-first correction and succeeds
at making pages shorter and smaller. It does not rescue two-scalar line encoding
on these eight-vector SIFT pages. Preserve the constrained packer and diagnostics
as opt-in ablations; independent quantized centers remain the leading measured
page-pruning representation. Do not infer an SSD speedup, global impossibility,
held-out generalization, or CLIP superiority from this result.

Reproduction and detailed algorithm contracts: `docs/COMPACT_SEGMENTS.md` and
`scripts/qualify_compact.py`. Keep the downloaded artifact beyond GitHub's 30-day
retention window.
