# Compactness-constrained short-segment qualification

## Question

Can a line summary become competitive when page construction explicitly protects
nearest-neighbor locality instead of grouping vectors primarily by collinearity?
This is a local heuristic experiment, not a globally shortest-segment algorithm.

## Construction

Keep the original IVF assignments, page count, and eight-vector page capacity.
Start separately from the frozen coordinate-GeoPack layout for each variant.
The original 128-dimensional FP32 vectors are neither quantized nor altered.
Only vector membership within existing IVF lists changes.

Use pools of at most eight full pages (64 points). A seven-round matching schedule
visits every pair of pages inside each pool once. Partial tail pages stay fixed.
For each pair's 16 points, propose balanced eight/eight splits from:

* consecutive eight-point axial windows along both current least-squares lines;
* assignments using joint axial/residual cost, not perpendicular distance alone;
* full-space center proximity and eight-nearest-to-seed subsets.

Every proposed page must obey its own fixed full-space diameter bound before it
can be shortlisted. The bound is `(1+epsilon)*original_page_diameter`; epsilon is
0 or 0.05. It does not increase after each accepted move. Thus repeated updates
cannot accumulate uncontrolled slack. The metric is full-space Euclidean distance,
not merely truncated PCA distance. Every final page is independently audited.

Rank feasible proposals relative to the current line models, freely refit the best
four balanced pairs, and accept only a strict decrease in the pair's total
`L^2 + 4*r_max^2`. L and r_max are measured in PCA64. This tube objective is a
compactness surrogate: the projected page diameter is bounded above by
`sqrt(L^2 + 4*r_max^2)`. It does NOT force either L or r_max to decrease on every
page and does NOT imply better query locality. Lines are least-squares fits, not
minimax-optimal fits for this objective. The full-space diameter constraint and
the page-count/IVF invariants remain the hard contracts.

Rebuild existing Line64 scalar-shell and scalar-ball summaries and independent
PCA18/8-bit and PCA64/4-bit summaries after packing. Count all code, scale and
radius bytes. Line geometry costs 172 B/page, PCA18 costs 176 B/page, and PCA64
four-bit costs 288 B/page. The common radial interval costs another 8 B/page.
No-filter runs in this harness load legacy metadata and are not minimal-RAM IVF.

## Controls

Four layouts: original GeoPack; previous residual-first pool64; strict compact
segments; compact segments allowing 5% full-space diameter growth per page slot.
Five execution settings per layout: no-filter whole-list scan, two independent
center encodings, and both line bounds. All filtered settings use the same
64-page windows and zero-gap coalescing; none uses any oracle threshold.

The canonical run keeps the previous seed=12345, nlist=1024, nprobe=64, k=10,
64 development queries and independent learning-file PCA. It repeats old-layout
baselines to verify continuity. No held-out conclusion, SSD speedup, or CLIP
comparison is established by this RAM-replay qualification.

## Layout versus representation

For each query, compute the final exhaustive IVF-candidate kth radius solely for
OFFLINE diagnostics. Count pages containing actual points at that radius and at
1.1x/1.25x the radius. This is a layout metric, not a query-planning input or a
formal lower bound on every conceivable search algorithm's reads.

For each summary, evaluate its bound at that same final radius after online
search. Verify that every actual in-radius page is admitted. Decompose counts as:

`online_reads = true_in_radius_pages + false_positive_pages_at_final_radius +
                extra_reads_above_final_radius_admission`.

The last term measures effects of starting with looser thresholds and genuine
seed reads under the chosen execution order, with no gap bridging in this run.
The decomposition is diagnostic, not an SSD-latency model.

Report true-page counts separately from page diameter. Small diameter is a
query-independent proxy and does not guarantee that co-accessed neighbors share
a page. Retain per-page diameters, segment spans, worst residuals, tube scores,
and actual per-query counts so these effects can be examined independently.

## Reproduction

```bash
python scripts/fetch_sift1m.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/qualify_compact.py \
  --work /path/to/empty/work --out /path/to/empty/results --queries 64
```

Use `.github/workflows/compact.yml` on the trusted experiment branch or manually
on main. It validates native direct I/O and io_uring but the qualification itself
uses RAM replay. One job, limited CPU threads, and no root/system configuration
changes. Source and logs are archived before this run's temporary payload copies
are removed. Preserve artifacts beyond the 30-day GitHub retention window.
