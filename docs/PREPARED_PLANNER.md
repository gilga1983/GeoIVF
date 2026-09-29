# Prepared, fused query planner

The optional `--selection prepared` path targets repeated per-window overhead,
not representation quality. It supports independent PCA ball sidecars at 4/6/8
bits and the unfiltered control. Other shapes remain on existing paths.

## Contract

- Preserve IVF assignments, within-list page ordering, scalar codes, measured
  error radii, radial intervals, initial seeding, planning windows, gap bridging,
  maximum extent size, full FP32 payload, and exact FP64 top-k kernel.
- Create a query-local context. Copy the query, calculate its complete projected
  coordinates and the existing numerical guard exactly once. Calculate its
  distance to each visited IVF centroid once, using the same norm reduction.
- Borrow packed index arrays; retain Python references for the context lifetime.
  Shapes, dtypes and contiguity are checked when preparing the context.
- Pass a contiguous [first,stop) page range directly to the native kernel, not
  an allocated/arbitrarily sorted page-ID array.
- Fuse radial filtering, threshold-aware ball rejection and extent coalescing
  in one C call. Coordinate sums remain sequential FP64 with no FMA contraction
  or fast-math. Sixteen-coordinate partial checks and page early acceptance use
  the same squared predicate as the previous native adaptive implementation.
- Specialize unpacking at compile time for 4/6/8 bits. Eight-bit scalar codes
  are loaded directly. Do not retain an expanded center table or new metadata.
- Reuse query-local mask and extent buffers. Return ordinary extent pairs to
  the existing reader. Preserve canonical completed-stage scan order.
- The context does not consult or populate CellIndex's mutable `_query` cache.
  Two contexts can represent different queries without donating cached work.
  Index arrays must remain immutable while a query context exists.

Scratch array storage is approximately 17*W + 2*D*8 bytes for a filtered query,
where W is the largest processed window and D the input dimension. For W=64,
D=128 this is 3136 bytes. Python objects and the at-most-nprobe radial cache are
additional. Scratch can grow for a larger seed window or whole-list processing,
but is released after the query. The shared directory RAM is unchanged.

## API and CLI

Build with `bash scripts/build_prepared.sh` (also builds existing kernels).
For the asynchronous reader, retain `bash scripts/setup_uring.sh`.

Use `search(..., selection='prepared', scan='native')`, or add
`--selection prepared --scan native --shape ball --summary SIDE_CAR` to the
existing CLI. For real I/O add `--backend uring --direct --pooled`.
An unfiltered prepared control uses `--filter none` and needs no sidecar.
The defaults are unchanged. All old controls remain executable.

## Validation and campaign

New tests cover all supported precisions, none/balls/radial/combined filtering,
partial tail pages, multiple windows and gap/extent limits, exact staged plans,
independent query contexts, input mutation, invalid options, and floating-point
threshold boundaries compared with the existing squared selection predicate.
Sqrt-based full lower bounds can round differently at a threshold; this is not
a claim of universal bitwise equivalence across all numerical implementations.

The canonical campaign is the existing `scripts/qualify_speed.py --prepared`.
It compares unfiltered native IVF, old full and adaptive ball execution, and
prepared ball execution at six/eight bits; includes a prepared unfiltered arm.
All arms use the SAME native exact scan and reader. Direct reads remain
O_DIRECT + io_uring at depth 16; real thresholds are learned from fetched data.

For each correctness query, compare IDs and SHA256 of the COMPLETE staged read
plan, including request lengths and stage boundaries. Also compare every page,
byte, request, stage and exact-distance count. Audit rejections after search.
Timing uses fresh routing and fresh query preparation inside each timed search.
Results are validated outside the timed interval. Randomize arm/query order in
every repeated round; retain the existing host file lock and process affinity.

This isolates planner engineering, not a change in precision, page organization
or ANN recall. It does not measure independent NVMe commands, steady-state QPS,
held-out generalization or superiority to Faiss/CLIP. Dataset/oracle memory in
the campaign is not deployment RAM. Compare old/new arms in the SAME campaign,
not raw times from a previous session with a different cohort or machine load.
