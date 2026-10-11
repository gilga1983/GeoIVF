# NavHints destination-memory investigation: successful fixed-payload and macro-region results

**Date:** October 11, 2026. Both experimental branches and successful Actions artifacts are separate from `DemmandAwareANN/main`. No approved PVLDB prose has been edited.

## Source and provenance

1. **Equal-budget tradeoff:** [successful GeoIVF run #38120556463](https://github.com/gilga1983/GeoIVF/actions/runs/38120556463), artifact #11696380166, branch `experiment/navhints-fixed-memory-tradeoff-20261011`, head `18b43f5237703ab54760a3e44302bd18a8cd630f`. The initial [failed run #38120315371](https://github.com/gilga1983/GeoIVF/actions/runs/38120315371) had an overly broad validation condition for the deliberately oversized reference; this was fixed before the successful run. Full numeric source: `results/navhints-memory-tradeoff.json`.
2. **Macro-regions:** [successful GeoIVF run #38120616752](https://github.com/gilga1983/GeoIVF/actions/runs/38120616752), artifact #11695664590, branch `experiment/navhints-macroregion-cache-20261011`, head `355d515d912cf6bc17fad5b210ccd9376b79f9c3`. Full numeric source: `results/macroregion-diagnostic.json`.
3. **Author-aligned CatapultDB:** [validated same-SSD comparison #38111405896](https://github.com/gilga1983/GeoIVF/actions/runs/38111405896), artifact #11692138871, original Core versus author-aligned CatapultDB-on-DiskANN, three repetitions and three hash seeds.
4. **Next same-session test:** [#38127358063](https://github.com/gilga1983/GeoIVF/actions/runs/38127358063), branch `experiment/navhints-optimal-core-vs-catapult-20261011`, compares old Core, new equal-budget Core, and corrected CatapultDB in one epoch. **Pending** at note creation; no unverified victory claim.

## 1. Equal accounted routing-state payload: 102,940 bytes

One pinned Microsoft DiskANN SSD backend, PubMed1M / MedCPT (1M x 768, inner product), K=10, beam=8. Offline junction directory derived only from disjoint first 5K training traces. Online cache starts empty and learns strictly causally from 4K warm-up requests; the last 1K are measured. Nine L values: {20,40,80,120,160,200,240,280,320}. Report logical SSD reads/query at matched Recall@10 without extrapolation.

With 512 fixed coarse cells, static directory payload is **36,892 + 4B bytes**, B = count of learned junction IDs. Winner cache payload is **4C bytes**, C = recent winner-ID capacity. Thus B+C=16,512 ensures equal **102,940-byte accounted payload**. This is not equality of process RSS, hash-set or allocator metadata.

| Junctions B | Destinations C | Extra starts K | Accounted payload | Mean read saving versus original Core |
|---:|---:|---:|---:|---:|
| 16,000 | 512 | 1 | 102,940 B | control |
| 16,000 | 512 | 4 | 102,940 B | 2.09% |
| 15,488 | 1,024 | 4 | 102,940 B | 4.22% |
| 14,464 | 2,048 | 2 | 102,940 B | 2.38% |
| 14,464 | 2,048 | 4 | 102,940 B | 6.00% |
| **14,464** | **2,048** | **8** | **102,940 B** | **7.42%** |
| 12,416 | 4,096 | 4 | 102,940 B | 5.94% |
| 8,320 | 8,192 | 4 | 102,940 B | 5.92% |
| 16,000 | 2,048 | 4 | **109,084 B (extra)** | 6.09% |

Means are equal-weight descriptive means across **eight mutually reachable recall targets** (baseline L=40 through L=320). The baseline L=20 recall is **not reachable** by these candidate curves; it is not included in the headline mean. This run uses one deterministic query sequence and cannot support a statistical confidence claim.

The leading equal-payload **B14,464 + C2,048 + K8** saves 9.44% SSD reads near the baseline 47.39%-recall target (L=40), 7.03% near 70.17% (L=160), and 4.85% near 79.81% (L=320). This is a substantial improvement at high recall, where the original Core was nearly tied with Catapult.

At K=4, shrinking 16,000 to 14,464 junctions costs only ~0.09 percentage points in the eight-anchor mean (6.09% → 6.00%). The extra 1,536 cache slots are therefore close to free on this workload in terms of static junction quality. K=8 scores **no new cached destinations**, since the implementation already PQ-ranks the ten best of the same cache; it merely offers more of those scored results as search starts. K=8 does increase *fixed-L* I/O versus K=4, but its recall gain makes its *matched-recall* I/O frontier superior across the measured common-recall region.

**Important occupancy/transfer limitation:** During the full 5K held-out replay, C2,048 never evicts; the high-recall K8 arm records **1,986 unique successes**, 3,014 duplicate successes, and zero evictions at L=160. C4,096 and C8,192 similarly do not fill, explaining why extra capacity alone provides little benefit. Original C512 records 2,649 evictions at L=160. Longer and more diverse histories are needed to learn the right steady-state eviction policy.

## 2. 16–256 learned macro query regions

The second successful run keeps the **same 16,000-junction/512-coarse-cell router**, same C2,048 and K4 in all arms. Only *destination lookup* changes: train-only spherical k-means groups its 512 existing coarse medoid embeddings into G=16/32/64/128/256 macro-regions. A 1,040-byte lookup map translates PQ-selected coarse cells to a query-local winner history. No additional full-vector LSH query hash is evaluated at runtime.

At L=160:

| History lookup | Recall@10 | Logical reads/query | Reported CPU/query |
|---|---:|---:|---:|
| **Global FIFO2048** | **71.17%** | **162.63** | 235.1 μs |
| Macro16, probe 4 coarse cells | 71.17% | 162.85 | 211.6 μs |
| Macro32, probe 4 | 71.17% | 162.83 | 203.4 μs |
| Macro64, probe 4 | 71.18% | 162.94 | 202.3 μs |
| Macro128, probe 4 | 71.16% | 163.02 | **197.7 μs** |
| 512-region LRU, probe 4 | 71.16% | 163.22 | 197.6 μs |

At L=80, Macro128/probe4 uses 83.69 reads/query and ~114.4 μs reported CPU vs global 83.31 reads and 149.5 μs CPU. At L=320 Macro128/probe4 uses 322.47 reads and 361.1 μs CPU vs global 322.07 and 395.8 μs.

**Interpretation:** Global FIFO remains ahead in SSD I/O, but macro-region views can reduce reported CPU residual by 10–25% at modest extra I/O and nearly unchanged recall. This is a tradeoff, not an unconditional win. Macro map, LRU timestamps, Vecs and allocation costs are currently additional to the nominal C×4 ID payload. The correctness-first prototype also scans metadata to find the oldest item in a full cache, which is not a deployable steady-state implementation. Do not claim an optimized or total-RAM-fair latency result from this pilot.

## Scientific recommendation

**Primary Core candidate:** simple global unique FIFO with **14,464 trained junctions + Recent2,048 and eight PQ-ranked additional starts**, at 102,940 bytes of accounted routing payload. It currently dominates the original Core's matched-recall I/O frontier across the eight qualifying targets. Preserve original Core for direct attribution.

**Secondary CPU-oriented candidate:** macro-region destination lookup, ideally G=32/64/128 with four nearest coarse probes. Carry only if it survives actual RSS/accounting and routing-time measurements.

**Before updating approved manuscript:** perform a same-session author-aligned Catapult head-to-head with the new equal-payload Core; test on Coveo and BigANN; extend the query stream to drive C2,048 into eviction; measure actual auxiliary heap/RSS and end-to-end latency with post-query feedback included in both arms. None of these successful one-stream pilots independently establishes cross-workload generality.
