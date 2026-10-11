# Why CatapultDB catches NavHints at high recall: evidence and diagnostic

Source of measured curves: successful, author-aligned SSD head-to-head [GeoIVF #38111405896](https://github.com/gilga1983/GeoIVF/actions/runs/38111405896), artifact #11692138871, `results/navhints-vs-author-aligned-catapult.json`. Same pinned Microsoft DiskANN, PubMed1M/MedCPT, identical SSD/index/query stream, 5K historical training / 4K online warm-up / 1K measured, 3 repetitions and 3 Catapult seeds. K=10, I/O beam width=8; RAM-only NavHints Core (102,940 B) versus author-aligned Catapult LSH/LRU SSD adaptation (106,496 B). The original Catapult implementation is an in-memory engine; this is a controlled *routing-policy* comparison.

## What is observed

At the **same search width L**, NavHints consistently requests around six to seven fewer logical SSD reads/query. Catapult nonetheless has slightly *higher recall at the same L*. The latter allows interpolation to a lower Catapult L when recall is matched; this is the source of the disappearing matched-recall advantage.

| Nav L | Nav recall (%) | Cat recall (%) at same L | Nav read | Cat read at same L | Same-L Cat read excess | Cat L to match Nav recall (interpolated) |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 29.31 | 29.86 | 19.85 | 26.61 | 6.76 | 11.56 |
| 20 | 36.75 | 37.61 | 27.50 | 34.53 | 7.03 | 19.11 |
| 24 | 39.34 | 40.28 | 31.43 | 38.46 | 7.02 | 22.60 |
| 40 | 47.39 | 48.84 | 45.94 | 53.17 | 7.23 | 37.30 |
| 80 | 58.72 | 60.14 | 84.02 | 91.19 | 7.17 | 74.96 |
| 160 | 70.17 | 71.06 | 163.05 | 169.61 | 6.56 | 153.46 |
| 320 | 79.81 | 80.28 | 322.36 | 328.35 | 5.99 | 311.84 |

By L=320, Catapult's approximately 0.47 percentage-point recall advantage permits an estimated **8.16-point decrease in search width**, compensating for its six additional same-L reads. This accounts *numerically* for the observed nearly flat or reversed difference at matched recall. The estimate interpolates between L=160 and L=320 and therefore is **sensitive to the unmeasured curve shape at high L**; dedicated dense L points are needed before claiming a robust crossover.

## What could produce Catapult's small recall advantage

1. **Candidate breadth:** Catapult's queried LSH bucket contains ~63 recent successful destinations on average during measurement, versus NavHints' one learned junction plus a single selected Recent512 destination. Catapult *offers* ~63 bucket IDs; that is not the number expanded. The pinned DiskANN benchmark overrides `num_starting_points()` to report 1 on both integrations, yielding a fixed priority queue of ~L+1 entries. Therefore at L=20, at most ~21 candidates initially survive, while at L>=80 substantially more of the offered seeds can fit.
2. **Conditional history:** Catapult has 256 query-region LRU buckets of up to 80 winner IDs, and queries use the bucket selected by random-hyperplane LSH. NavHints' Recent512 value cache is global, and it injects only the best PQ-scored destination; its ~16K static hint entries are junctions rather than distinct recent successful outcomes. Catapult may therefore cover more distinct regions/destinations at high L despite comparable total routing payload.
3. **Ranking and overhead:** NavHints spends ~900--1,000 additional counted PQ comparisons per query on coarse/fine junction selection and the 512-value scan (not all Catapult LSH dot-product operations are counted by this metric). At fixed L, the NavHints I/O advantage is nearly constant rather than growing with L. We should diagnose *routing-recall efficiency* rather than merely the I/O count per query.

**These are candidate mechanisms, not demonstrated causal explanations.** In particular, Catapult's region-conditioned history and its large offered-start set are confounded in the existing head-to-head.

## Mechanism-only diagnostics launched

Isolated self-hosted [run #38115574578](https://github.com/gilga1983/GeoIVF/actions/runs/38115574578) evaluates:

- NavHints Recent512 injection K=1/2/4/8/10, keeping its **exact same 512 PQ scores, static 16K directory, and auxiliary memory**. K=1 preserves the approved original selector. If K>1 improves same-L recall and matched-recall I/O at high L, limiting injected candidates is contributing to the apparent Catapult advantage.
- Catapult bucket entry cap = 1/4/16/all, choosing most-recent entries when capped and retaining **the same trained LRU snapshot** within each L. This tests whether dozens of offered starts are necessary for Catapult's fixed-L recall improvement. Unlimited reproduces the author-aligned adaptation.
- Unmodified DiskANN baseline on the same final 1K held-out queries, for an absolute reference curve.
- Widths L=20/40/80/160/320. Strict 4K causal warm-up, final 1K measurement, one representative Catapult hash seed. This is deliberately an exploratory ablation **without statistical replication**; follow successful hypotheses with 3 hash seeds × repeated traces, and use a denser high-L grid before publication.

The diagnostic branches never modify the approved paper or main branch. Do not upgrade the current results to publication claims before successful tests and verification.
