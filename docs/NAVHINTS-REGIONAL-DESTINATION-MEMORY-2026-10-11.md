# NavHints: regional destination memory (exploratory)

## Motivation

NavHints Core uses an already-learned 16K graph-junction directory indexed by a 512-cell coarse PQ codebook, plus a global **Recent512** pool of successful result IDs. The latter reserves exactly 512 × 4 = 2,048 bytes of ID payload. Every request scores the entire cache with its existing PQ LUT and emits only its best result as an additional starting point. The update policy is a **unique FIFO**: an already-stored successful destination is *not refreshed*. It may be evicted despite many subsequent successful accesses.

CatapultDB combines random-hyperplane LSH query partitioning with per-bucket LRU winner histories and offers many stored destinations as query starts. In the source-verified same-SSD study [#38111405896](https://github.com/gilga1983/GeoIVF/actions/runs/38111405896), this broad candidate set has slightly higher recall at the same L, even though NavHints uses ~6–7 fewer logical SSD reads/query. This recall difference explains high-L matched-recall convergence; it does NOT yet identify whether the cause is history size, recency policy, query-region conditioning, or seed width.

## Design space (isolated test; paper implementation unchanged)

Three orthogonal axes:

1. **Time:** unchanged global unique FIFO; global **LRU** that refreshes successful hits; demand-weighted **region-local LRU**.
2. **Memory:** Recent512 (2KB ID payload), Recent2048 (8KB), Recent8192 (32KB). All preserve the existing 16K junction map, so the larger configurations are deliberately *not* equal-total-RAM to baseline or Catapult.
3. **Spatial locality / candidate breadth:** local histories keyed by nearest coarse Hint-IVF region, already scored during normal routing; retrieve from nearest 1 or 4 regions. Optionally emit up to 4 best PQ-scored cached destinations rather than 1. No additional LSH projections, RAM-resident full vectors, new graph edges, or writes.

For the regional prototype, each successful destination is stored under the **nearest coarse cell of that query**, not under the eventual result vector's nearest cell. This is conditional-on-query history, not merely region-labeled hot vectors. One global budget covers all 512 region buckets; the **globally least-recent (region, winner)** record is evicted when the budget fills. Popular regions therefore claim more slots and underused regions do not reserve unused fixed capacity. A stored destination may have separate entries in several query regions.

The current correctness-first prototype holds an insertion stamp and a `Vec` per region; its actual resident memory exceeds the 4-byte-per-ID payload used in configuration names. It scans timestamps to find the global oldest at eviction, and allocates/scans candidate slices on search. All latency claims must wait for this metadata to be compacted. Record logical read/recall behavior now; **do not misrepresent the pilot as equal-RAM or production-grade runtime**.

## Pilot experiment

Self-hosted [run #38116221438](https://github.com/gilga1983/GeoIVF/actions/runs/38116221438); isolated branch `experiment/navhints-regional-history-20261011`. Existing source-affinity fingerprint at K=10, beam=8, frozen 1M PubMed graph, 5K prior history, 4K causal warm-up and final 1K measured queries. Search widths L=40,160,320, with no persistent writes. Original FIFO512 is retained as exact control.

Arms:
- FIFO512, global LRU512, region-LRU512.
- FIFO2048, global LRU2048, region-LRU2048 from nearest 1 or 4 cells.
- Global LRU8192, region-LRU8192 from nearest 1 or 4 cells.
- Region-LRU2048/8192 with four nearby cells and the best four cached starts injected.

This is a **single-replay exploratory mechanism pilot**, not a statistical result. It cross-checks original FIFO512 recall and logical reads with successful #38111405896 at all three widths. The earlier [candidate-width diagnostic #38115574578](https://github.com/gilga1983/GeoIVF/actions/runs/38115574578) is an independent experiment that tests the starting-point breadth hypothesis.

## Follow-up acceptance criteria

1. Verify that global FIFO512 remains bit-identical in recall/reads. Reject any new method that changes the frozen graph or query frontier constraints.
2. Separate *better recall at same L* from *lower reads at matched recall*. Do not extrapolate outside the common recall interval. Measure per-width CPU work because a large global cache may be expensive to PQ-score.
3. If regional behavior wins, replace linear oldest-record scans with a compact global LRU index and pack region IDs with bounded metadata. Measure actual allocations/RSS and per-query CPU time including feedback updates.
4. **Strict-memory test:** redistribute the total ~103KB by reducing the 16K static junction directory and increasing online destination-state capacity. Compare Pareto frontiers at exactly matched **total** auxiliary memory, and check whether online metadata erases the advantage.
5. Re-test on production-chronological Coveo and weak-reuse BigANN; the PubMed effect alone does not establish generality. Optionally test an additional Catapult LSH-bucket hybrid when the learned-region signal is weak.

Scientific working hypothesis: coarse graph routing regions are a better, *free* query-locality key than an independent random LSH projection, because their PQ scores are already computed. This is falsifiable; adjacent-cell retrieval, regional skew, and LRU eviction must earn their complexity empirically.

**No edits to the approved PVLDB paper.**


## Observed pilot results (verified completed run #38116221438)

[Successful self-hosted regional-memory run](https://github.com/gilga1983/GeoIVF/actions/runs/38116221438), artifact `navhints-regional-history-pilot` #11693617781. Separate successful [starting-candidate-width run](https://github.com/gilga1983/GeoIVF/actions/runs/38115574578), artifact #11694300439, with one representative Catapult seed. Both run on the same pinned SSD graph and causal PubMed replay.

**Original FIFO512, k=1 control is exactly unchanged** in native logical reads and Recall@10 at every measured L (40,160,320). Pilot read costs and recall:

| Policy | L=40 recall/read | L=160 recall/read | L=320 recall/read |
|---|---:|---:|---:|
| Original global FIFO512, one start | 47.39% / 45.935 | 70.17% / 163.053 | 79.81% / 322.364 |
| Global LRU512, one start | 47.52% / 45.916 | 70.22% / 163.013 | 79.84% / 322.358 |
| Global LRU2048, one start | 47.44% / 45.612 | 70.16% / 162.889 | 79.71% / 322.206 |
| Learned-region LRU2048, 4 queried cells, one start | 47.51% / 45.700 | 70.20% / 162.916 | 79.77% / 322.246 |
| Learned-region LRU2048, 4 queried cells, four starts | **49.19% / 45.038** | **71.16% / 163.216** | **80.22% / 322.664** |
| Learned-region LRU8192, 4 cells, four starts | 49.00% / 44.833 | 71.19% / 163.081 | 80.21% / 322.564 |

The experiments show three distinct facts. (1) Enlarging the **global** cache beyond approximately 2K IDs adds little because fewer than 2.1K distinct winners are present in the 5K-query replay; 8K capacity remains mostly empty. (2) Changing global FIFO to LRU yields only minor fixed-L recall/read improvements. (3) **Combining region-conditioned memory with four starts** yields +1.80, +0.99 and +0.41 recall percentage points versus original Core, with nearly identical SSD reads. A *single* region-conditioned start alone has little advantage.

In the independent candidate-width ablation, four starts from the **global FIFO512** produce 48.41%/46.822, 70.68%/164.390, 80.03%/323.664 at L=40/160/320. The regional-2048+four-start variant gives better recall *and* fewer reads at each of those widths, supporting (but not proving independently) that query-regional conditioning adds value beyond candidate width.

**Crucial limitations:** Regional 2,048-ID *payload* is 8KB, compared with 2KB for original Core, increasing the accounted payload from 102,940 to 109,084 bytes. The prototype additionally holds a timestamp per regional record and hundreds of Vec/container headers; **real RAM and CPU overhead are not equal**. The region-8192 variant uses even more ID RAM but actually stores only roughly 2.5K (region, winner) records after warm-up and evaluation. Latency fluctuates strongly and the correctness-first oldest-record search is O(B); do not claim a latency improvement. These are single-stream exploratory findings, and high-L matched-recall claims require a denser L grid and repeated measurements. A cross-run comparison against original author-aligned Catapult (seed 1) suggests the regional 2048+4 configuration now meets or slightly exceeds its fixed-L recall while saving logical reads; do **not** treat this as an equal-RAM, repeated Catapult head-to-head.

### Next isolation: matched capacity and matched candidate count

A same-binary, same-trace [factorial run #38118536829](https://github.com/gilga1983/GeoIVF/actions/runs/38118536829), branch `experiment/navhints-regional-factorial-20261011`, was launched to compare global FIFO/LRU512/2048 versus region-local 512/2048 at **exactly the same injected start count (K=4)**. Also varies regional query selection (top-1 versus top-4) and checks high-L behavior. This will help distinguish candidate breadth, cache size and query-conditioning as separate effects. The run's status is not implied by this note.
