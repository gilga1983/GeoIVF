# PCA and residual-norm qualification results, 2026-09-29

Implementation commit: `f21f0826138e7589088911577d7807cb8adad59c`.
Successful Actions run: `36524087336`; artifact ID: `11013921836`.
Artifact SHA256: `7705bc25bdd77c83c2a8fee8867756448ce191729165c3cd51d70cfabf0d6ebb`.

## Scope and verified checks

Canonical TexMex SIFT1M, full 1M base vectors, 128D FP32 payload unchanged, 4096-byte pages, eight vectors/page, IVF nlist=1024, k=10, construction seed=12345. Main comparisons use 64 development queries and nprobe=16/64. Scheduling diagnostics use the first 16 of those same development queries. No held-out evaluation or SSD-latency claim is made.

All representations use exactly the same GeoPack payload. Its SHA256 is `8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`, identical to the preceding qualification. Packing still uses the original 16-coordinate layout, not PCA. All candidate lists are preassigned and identical across representations.

PCA was fitted only on the canonical 100000-vector learning file. Quantizer ranges use indexed vectors, not development queries. Its complete FP64 transformation was norm-bounded before use. Independent SVD inspection of the archived matrix gives operator norm 0.9999999999083635; the stored normalization divisor is 1.000000000091637.

All 60 runner tests passed with zero skips, including the existing O_DIRECT/io_uring tests and 20 new projection tests. All 1360 configuration-query comparisons matched the independent FP64 IVF candidate-set oracle. There are 64 distinct queries, not 1360 independent queries. The 31 configurations comprise 18 representation/control settings on 64 queries and 13 scheduling settings on 16 queries.

168166 rejected-page decisions were checked against actual original vectors after search, with no unsafe decision found. Audits cover the first two queries in each configuration and include repeated pages across configurations. They are not 168166 distinct queries/pages. Per-query CSV row counts, query IDs, all aggregate means, read_bytes=4096*read_pages, and read_pages=selected_pages+gap_pages_read were independently reconciled after downloading the artifact.

## Main result at nprobe=64

All methods have Recall@10 = 0.99375 on the same 64 queries. Values below are requested pages/extents from RAM replay, NOT observed SSD traffic. Geometry bytes exclude the common eight-byte radial interval per page and shared projection metadata. Total directory arrays are reported separately.

| Representation | Geometry B/page | Pages/query | Aggregate page reduction | Median per-query reduction | Extents/query |
|---|---:|---:|---:|---:|---:|
| none-whole-list | not minimal RAM | 8394.7969 | 0.0000% | 0.0000% | 65.0938 |
| none-seeded-w64 | not minimal RAM | 8394.7969 | 0.0000% | 0.0000% | 163.6875 |
| coordinates16-u8 | 160 | 8155.9531 | 2.8451% | 0.0000% | 224.3125 |
| pca16-u8 | 160 | 6173.8750 | 26.4559% | 11.7122% | 595.9375 |
| pca16-f32 | 544 | 6078.5781 | 27.5911% | 13.1811% | 611.8281 |
| pca32-u8 | 288 | 2455.9219 | 70.7447% | 72.5192% | 822.5938 |
| pca64-u8 | 544 | 297.9844 | 96.4504% | 97.0974% | 156.6406 |
| pca8-tail-u8 | 160 | 7439.1875 | 11.3834% | 0.3701% | 353.8125 |
| pca16-tail-u8 | 224 | 5915.3594 | 29.5354% | 14.9771% | 654.5312 |

Both the baseline and PCA runs load summary arrays in this research harness, so the no-filter allocation is not a minimal-RAM IVF implementation. No-filter whole-list reads and seeded-window reads have identical page counts but different request/stage counts.

At nprobe=16, common Recall@10 is 0.9234375. Unfiltered pages/query = 2156.859375. PCA16 uint8 reads 1905.484375 (11.6547% reduction), PCA32 uint8 reads 1157.90625 (46.3152%), and PCA64 uint8 reads 255.9375 (88.1338%).

## Interpretation

PCA16 improves substantially over selecting 16 original coordinates at equal per-page geometry bytes: 2.8451% versus 26.4559% aggregate page reduction on this expanded sample. The earlier original-coordinate figure was 1.9418% on 16 queries; the difference is sample expansion, not a changed baseline.

All 64 queries save at least one page with PCA16, PCA32 and PCA64. The medians are 11.7122%, 72.5192% and 97.0974%, respectively. The largest individual query contributes about 6.01%, 2.40% and 2.00% of each method's total saved pages. Thus the PCA64 result is not driven by the single exceptional query seen in the original-coordinate pilot.

Replacing uint8 PCA16 centers with FP32 improves pruning only from 26.4559% to 27.5911%, while increasing geometry from 160 to 544 bytes/page. At the same 544-byte budget, uint8 PCA64 instead yields 96.4504%. This favors retaining more dimensions rather than increasing precision for these configurations.

The residual norm bound is valid but is not the best use of bytes in this sweep. Equal-budget PCA8+tail gives 11.3834%, versus PCA16 at 26.4559%. PCA16+tail gives 29.5354% but costs 224 instead of 160 geometry bytes/page. This does not rule out other residual encodings.

## Memory and offline headroom

PCA16 geometry is 20071200 bytes; PCA32 is 36128160; PCA64 is 68242080. All have 1003560 radial-interval bytes. Complete array directories are 31075730 bytes (29.64 MiB), 47132946 bytes (44.95 MiB), and 79247378 bytes (75.58 MiB), respectively. Python/runtime, routing copies and query transients are additional. The peak experiment RSS of about 3549 MiB includes base vectors, oracle data, all variants and projection scratch and is NOT deployment RAM.

The PCA64 geometric-plus-radial summary is 552 bytes per 4096-byte payload page, or 13.4766%, before other directory metadata. Its larger memory cost must not be hidden behind the same-recall comparison.

Ideal unquantized head-only proxies, evaluated OFFLINE at the final IVF kth radius, could reject 27.7011%, 73.9696%, and 98.2990% of pages at dimensions 16, 32, and 64. The full 128D diagnostic rejects 99.8896%. These are optimistic diagnostic counts, not an online algorithm or a wall-clock speedup. The online filter never receives an oracle threshold.

## Scheduling and fragmentation

The following use the same first 16 development queries; they must not be compared directly to 64-query aggregate means above. All have baseline 8448.9375 candidate pages/query.

| Configuration | Pages/query | Extents/query | Read stages/query |
|---|---:|---:|---:|
| Unfiltered whole-list | 8448.9375 | 65.0625 | 64.0000 |
| Unfiltered seeded, window64 | 8448.9375 | 164.5625 | 164.5625 |
| PCA64, window16, gap0 | 299.6250 | 178.5625 | 98.6875 |
| PCA64, window64, gap0 | 317.6875 | 167.5000 | 51.5625 |
| PCA64, window256, gap0 | 357.8125 | 156.6250 | 29.5000 |
| PCA64, whole-list window, gap0 | 358.7500 | 156.5625 | 29.2500 |
| PCA64, window64, gap1 | 349.8750 | 135.3125 | 51.5625 |
| PCA64, window64, gap2 | 386.8750 | 116.8125 | 51.5625 |

For PCA16 on these same 16 queries, moving from window16 to window256 changes pages only from 6534.0625 to 6534.6250, while extents fall from 883.3750 to 530.5000. Thus window overhead and genuine fragmentation are distinct. Gap2 with window64 reduces its extents to 260.1875 but increases requested pages to 6952.3125.

The common read plan remains a critical tradeoff. On the 64-query main comparison, PCA16 reduces pages 26.5% but increases extents from 65.1 to 595.9. PCA32 reduces pages 70.7% but raises extents to 822.6. PCA64 reduces pages 96.5% with 156.6 extents. These are application requests, not automatically physical NVMe commands. A real asynchronous I/O experiment is required before claiming speedup.

## Next gate and limits

The canonical pruning gate is now promising. The immediate next gates are native/common-kernel execution, calibrated asynchronous reads, upstream CLIP measurements, and a fixed-representation comparison of original/radial/GeoPack layouts. This sweep isolated representation quality on one frozen layout; it does not establish GeoPack's incremental benefit with PCA.

Only one canonical dataset, one IVF construction seed, 64 development queries, and RAM replay are covered. There is no measured SSD speedup, no CLIP/HIVF-CLIP result, no held-out test result, and no billion-scale claim. Floating-point safeguards are engineering measures rather than a formally verified numeric library.

Reproduction and derivation are in `docs/PROJECTION_QUALIFICATION.md`. The complete artifact includes query IDs/routes, the learned PCA matrix, per-query CSVs, offline diagnostics, aggregate JSON, dataset hashes, package versions, tests and logs. Its GitHub retention is 30 days; retain the downloaded archive as part of the research record.
