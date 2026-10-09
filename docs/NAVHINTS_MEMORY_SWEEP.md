# NavHints competitor RAM-scaling qualification

**Paper question:** how do established navigation and caching alternatives compare with NavHints at comparable auxiliary memory, and when given more RAM?

**Experiment:** [GitHub Actions workflow](../.github/workflows/navhints-memory-sweep.yml), self-hosted Linux x64 geoivf runner only. The evaluation is serialized with the frozen competitor cache-correction workflow using GitHub concurrency, and all SSD search runs acquire `~/.cache/geoivf/speed-device.lock`.

## Fixed matched-recall protocol

- Dataset: PubMed1M/MedCPT, same frozen 1M × 768 index, PQ data, NVMe SSD.
- Historical workload: 10K MedRAG-Zipf requests, first 5K for static training, next 4K to warm online state, final 1K to measure.
- Exactly the same held-out queries, ground truth, K=10, beam width 8, four search threads, and 12 values of search-list capacity L.
- Methods rotate ordering across three repetitions. Catapult uses seeds 0, 1 and 2.
- Report both measured SSD graph-record reads and latency at matched Recall@10, not the result at a conveniently chosen L. The anchored reference is Full NavHints with p=1/2.
- Every competitor result and its eligible recall anchors will be retained. Present the full sweep, not a post-hoc best configuration as a tuned model.
- All runtime route selection, candidate scoring, and cache lookup are included in the timed query path. All static methods receive the same 4K storage-warm-up queries.

## Memory-budget variants

All budgets count auxiliary/query-side payload and are approximate 1×/2×/4×/8×/16× of the **102,940-byte** NavHints core payload, excluding comparable container/allocator overhead. Persistent destination IDs live on disk, accounted separately.

| Tier | Hot full-node cache (nodes; estimated vector + 64 edge IDs) | DiskANN++-style QSEV (entries; exact bytes) | Catapult, 7 hash bits (capacity; reserved bytes) | Catapult, 8 hash bits (capacity; reserved bytes) |
|---|---|---|---|---|
| 1× | 30; 99,840 B | 32; 98,448 B | 160; 103,424 B | 80; 106,496 B |
| 2× | 60; 199,680 B | 64; 196,880 B | 360; 205,824 B | 180; 208,896 B |
| 4× | 120; 399,360 B | 128; 393,744 B | 760; 410,624 B | 380; 413,696 B |
| 8× | 240; 798,720 B | 256; 787,472 B | 1,600; 840,704 B | 780; 823,296 B |
| 16× | 480; 1,597,440 B | 512; 1,574,928 B | 3,200; 1,659,904 B | 1,580; 1,642,496 B |

The cache baseline uses the graph's full record vectors and adjacency, whereas QSEV stores full-precision entry vectors. Catapult's bucket capacity is a maximum/reservation, not necessarily filled; recorded snapshots report *actual* populated destination IDs as well as reserved payload. The reported QSEV selection includes full-precision scan time. QSEV here measures only DiskANN++'s entry-selector, not its full page-oriented graph layout. An additional native BFS-30 DiskANN cache is retained as a sanity control, while the scalable cache is trained on workload-hot graph vertices from the same history.

## Validation / guardrails

1. Rust cache patched to count **actual uncached graph-record loads**, not every requested expansion. The original pinned DiskANN benchmark attributed zero cache hits because both counters were derived from the same requested expansions.
2. Cache and BFS variants must report a positive cache hit percentage at L=160, strictly fewer SSD reads than baseline and unchanged recall. Otherwise the experiment fails.
3. Catapult snapshot training (first 5K), causal online warm-up (next 4K), and final measurement (1K) use separate state images per configuration and random seed.
4. Present method memory and CPU costs with performance; larger Catapult capacities can remain unfilled, which should be stated rather than represented as consumed RAM.
5. SSD device exclusivity is enforced by the shared `speed-device.lock`.
6. All raw per-L results and config manifests remain in the workflow artifact. The paper should not claim a memory scaling result until these validation checks pass.
