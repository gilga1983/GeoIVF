# NavHints destination-cache investigation, October 11, 2026

## Validated observations

The [author-aligned CatapultDB comparison](https://github.com/gilga1983/GeoIVF/actions/runs/38111405896) used the same pinned DiskANN SSD index and RAM-only NavHints Core at broadly matched auxiliary RAM. Our original 16K junction + global Recent512 cache saves 9.88% reported logical I/O across the seven shared Recall@10 anchors, but the relative advantage vanishes at high recall because Catapult has slightly better recall at each fixed L.

To isolate why, a [high-recall candidate-width experiment](https://github.com/gilga1983/GeoIVF/actions/runs/38115574578) tested 1/2/4/8/10 *already PQ-scored* cached results. More than one starting destination increases fixed-L recall, with diminishing returns after four. Another [global/regional factorial](https://github.com/gilga1983/GeoIVF/actions/runs/38118536829) separated cache capacity and candidate injection width:

| L | Original 16K+Recent512, K1 (recall, logical reads) | Global FIFO2048, K4 | 512-region LRU2048, K4 |
|---:|---:|---:|---:|
| 40 | 47.39%, 45.935 | 49.28%, 44.230 | 49.19%, 45.038 |
| 80 | 58.72%, 84.021 | 60.33%, 83.314 | 60.12%, 83.924 |
| 160 | 70.17%, 163.053 | 71.17%, 162.630 | 71.16%, 163.216 |
| 320 | 79.81%, 322.364 | 80.19%, 322.072 | 80.22%, 322.664 |

This supports an uncomplicated candidate: **larger global FIFO history, four PQ-ranked starts**. Global FIFO2048 and LRU2048 have identical search results because the 5K online sequence produces fewer than 2048 distinct successful destination IDs; neither policy evicts. That fact does **not** imply LRU never matters on longer or higher-diversity workloads.

The 512-region conditional LRU is not the I/O winner when cache capacity and seed width are matched. However, it reports ~40-50 microseconds lower CPU residual per query on this hardware because it scores a much smaller set of cached candidates. Those CPU counters require direct replication and are not isolated per-query routing-cycle measurements. The correctness-first regional LRU still incurs expensive O(B) global oldest-item scans at eviction, uses per-entry timestamps, and consumes extra container memory.

## Current follow-up I: strict auxiliary payload tradeoff

[Run #38120556463](https://github.com/gilga1983/GeoIVF/actions/runs/38120556463) is the recovery of the [initial harness failure #38120315371](https://github.com/gilga1983/GeoIVF/actions/runs/38120315371). All re-built static directories from the first attempt were successfully archived; the fix corrects validation that mistakenly rejected the deliberate extra-budget reference. The benchmark uses one frozen 5K historical training prefix, identical 4K causal warm-up / 1K measured queries, fixed graph/PQ/SSD, and RAM-only Core.

The original 16,000-junction/512-result configuration occupies 102,940 bytes of *accounted routing payload*. With 512 fixed coarse cells, directory byte size is `36,892 + 4×junction_count`. Thus every `B + C = 16,512` configuration uses exactly 102,940 bytes when B = learned junction IDs, C = online winner IDs.

Test matrix:
- B16,000 + C512, inject 1 and 4 starts.
- B15,488 + C1024, inject 4 starts.
- B14,464 + C2048, inject 2, 4, and 8 starts.
- B12,416 + C4096, inject 4 starts.
- B8,320 + C8192, inject 4 starts.
- B16,000 + C2048, inject 4 starts, **unrestricted extra-RAM reference**, not part of equal-payload claim.

Each B is independently clustered from only the frozen training trace, preserving 512 coarse regions and original clustering seed. Test L={20,40,80,120,160,200,240,280,320}; report measured recall/I/O points and interpolated equal-recall results, without extrapolation. The additional hash-set, scoring scratch, and allocator metadata means these are **equal accounted payload**, not yet equal resident-set size.

## Current follow-up II: macro-region conditional history

[Run #38120616752](https://github.com/gilga1983/GeoIVF/actions/runs/38120616752) keeps the frozen **same 16K-junction/512-coarse-cell router**. It only groups its 512 learned coarse medoid IDs into 16/32/64/128/256 macro query regions through offline 6-iteration spherical k-means on the original medoid embeddings. The final mapping is 1,040 bytes (`GHMAC001` plus 512 two-byte IDs), and the query uses *already PQ-scored* coarse cells to look up its macro history. No extra random LSH projections.

A global FIFO2048/K4 arm is the reference. Macro-region LRU2048/K4 arms query nearest 1 or 4 coarse-region buckets, compared with original 512-region LRU2048/K4 controls. Same causal history and L={40,80,160,320}. Compare recall, logical SSD reads, and CPU residuals, while explicitly counting mapping/region/timestamp memory overhead. The purpose is to find a locality index that preserves the global pool's recall with lower PQ-selection CPU cost, **not** to assert a smaller-RAM prototype.

## Next decision rule

1. Prefer the *simpler global FIFO* unless query-local macro grouping establishes a repeatable Pareto improvement in CPU at nearly unchanged matched-recall SSD reads. A larger cache alone should never be mistaken for stronger query-conditioned locality.
2. If B14,464+C2048/K4 beats original Core under identical accounted payload, take it to multi-workload and Starling portability tests. Include timing-boundary harmonization so all policy feedback work is measured.
3. Regardless of which policy wins, implement compact accounting before claiming equal total process RAM; measure per-query routing CPU directly rather than only subtracting SSD wait time from total latency.
4. Re-run with a much longer history and varied query supports. The current C2048 cache never evicts for global FIFO, so it cannot answer steady-state replacement questions or the lifetime of query-local advantages.

**All experiments and notes are in isolated GeoIVF branches; the approved PVLDB manuscript and DemmandAwareANN/main remain unchanged.**
