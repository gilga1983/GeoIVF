# RAM-first ranking and certified verification

This experiment keeps the original GeoPack pages, learning-file PCA64 transform,
eight-bit independent scalar codes, FP32 covering radii, IVF routing and FP32
SSD payload unchanged. It changes when full vectors are fetched.

## Query execution

1. Select the same nprobe IVF lists. A native streaming scan computes projected
   distances to the decoded singleton proxies and retains the best R distinct
   vector IDs in an O(R) max heap. Rank scores are approximations, NOT safe
   full-space thresholds. No original vector is read during this step.
2. Deduplicate shortlisted vector locations into physical pages. Coalesce within
   IVF lists and read these pages. Evaluate ALL original vectors on every read
   page using the same native FP64/SIMD top-k kernel as staged execution.
3. Approximate mode stops here. Exact reranking does not recover a neighbor whose
   page was omitted. Report global recall and agreement with the IVF oracle.
4. Certified mode also computes each selected-list page's conservative lower
   bound during the RAM scan. Combine its ball-union bound with the existing
   deterministic radial bound. After first-wave verification, let tau be the kth
   exact distance of fetched vectors. Read every remaining page with LB <= tau.
   This second read set is frozen BEFORE those reads start. No oracle threshold
   or confidence-calibrated cutoff is supplied to online search.

Why two waves suffice: omitted pages have LB > tau, and thus contain no vector
with distance <= tau. The kth distance cannot increase as verified candidates
are added. Fetching all unread pages not excluded by that certificate therefore
recovers the selected-list IVF answer. Strict rejection retains distance ties
for the existing ID tie-break. If the candidate set has fewer than k points,
the first shortlist contains all of them. Floating-point implementations retain
the existing conservative engineering guards, not formal interval arithmetic.

Two VERIFICATION WAVES do not imply two SSD commands. Each contains coalesced
extents and is processed at the reader's configured queue depth. Large waves
are split into <=16 MiB power-of-two buffer-reservation batches. Their page set
is not reconsidered between batches. Report dependent waves and actual reader
calls separately. No page is scanned twice, including gap pages already read.

## Implementation and overhead

`native/ram_rank.cpp` scans packed 4/6/8-bit coordinates without materializing
expanded centers. Approximate-only mode skips radius/lower-bound calculations.
The initial scorer uses squared projected-center distance, not a full-vector
reconstruction or a PQ ranker. Certified mode reuses these distances for bounds.

`geoivf/ramfirst.py` handles query projection, native ranking, page deduplication,
coalescing, bounded read batching and the optional second-wave certificate.
The persistent directory is unchanged. Query scratch includes O(R) ranked slots
and scores, O(number of candidate pages) page IDs and optional lower bounds,
visited-page bookkeeping and temporary extents. The research index still loads
the radius array in approximate mode so it has the same directory as controls;
no radius-memory saving is claimed by that mode.

Use `bash scripts/build_ramfirst.sh --uring` for native and asynchronous builds.
Call `search_ramfirst(index, query, reader, shortlist=64, certified=True)` on a
certified independent-ball CellIndex and the existing pooled reader. Existing
search defaults are unchanged; this is an opt-in experimental path.

## Qualification

`scripts/qualify_ramfirst.py` sweeps R in {16,32,64,128,256} in both modes, against
our leading staged baseline (PCA64/8-bit, prepared selection, SIMD scan, 256-page
windows, gap2). RAM-first starts with no gap bridging. All methods share the
same reader (rolling io_uring, O_DIRECT, queue depth16), routes and exact scanner.

The full canonical SIFT1M database is used. Development queries are the first
128 entries of permutation(seed=20260929), inside the original development pool.
A separate 256-query test uses permutation[1256:1512], excluding the entire old
1000-query development pool AND all 256 previously examined external-test queries.

Three randomized interleaved direct-I/O rounds on development select the fastest
approximate and certified configurations meeting 99% strict Recall@10. If none
meets the target, mark the family infeasible and carry its highest-recall point
as a diagnostic, not a successful setting. Freeze choices before any test timing.
Only these two choices and the staged baseline are evaluated on the new test set,
with three rounds each of direct I/O and RAM replay. No held-out retuning.

All exact-mode results must match the independent full-vector IVF candidate-set
oracle. Approximate results are explicitly allowed to differ and their loss is
reported. Repeated IDs/counts must agree across timing rounds. Save individual
neighbor arrays, strict global recall, IVF-reference recall, first/second-wave
pages, reader calls, extents, bytes and timing categories. Rejection audits run
after online search and never initialize the search threshold.

This is a same-engine architectural experiment, not a new DiskANN or CLIP run.
Do not compare earlier cross-campaign timings as an exactly matched SOTA test.
The host is shared, affinity is not exclusive, frequency/device caches are not
controlled and the small index fits RAM. O_DIRECT is genuine but not evidence
of larger-than-RAM scale. Campaign RSS includes the base/oracle and is not the
deployed directory footprint. Preserve downloaded artifacts beyond retention.
