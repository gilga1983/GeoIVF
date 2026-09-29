# External-baseline tradeoffs, 2026-09-29

## Evidence and scope

This report analyzes TWO ALREADY COMPLETED runner campaigns. It does not claim a new benchmark was launched during this analysis.

In-memory IVF/CLIP campaign:
- Workflow 36585468457, tested commit 8d34b875962cdc18250b706f3c037404f7b46d12 on experiment/external-ivf.
- Artifact 11041751897, external-ivf-36585468457.
- ZIP SHA256 c5225b02eaec0417163fb6912101a9684dcc36bbf8ff914a780eb6734d898607.
- Released CLIP revision 7f4fc84edffede0aa21fae6131ec7391ce99ab6f; bundled Faiss e8234e563f1ecef5f036e83c3cfee366d3f1fbca. Separate packaged Faiss control: 1.15.1.

Disk-provider campaign:
- Workflow 36585360305, tested commit 08b4b9ab7be37440ec8093444375a6ed5085c95a on experiment/external-disk.
- Artifact 11042460252, external-disk-36585360305.
- ZIP SHA256 d82ec00d9daad7db408816f7dfdaa5c71a84a44639082cba2320f1267b5238b4.
- Released Microsoft DiskANN revision fcf90534174cf29c78c9f13b4cccf1fcabff85f5, Rust DiskANN3 disk provider and benchmark.

Both use all 1,000,000 original 128D FP32 SIFT1M vectors. Development uses 128 queries; held-out uses the SAME 256 distinct queries in both campaigns. The held-out IDs are permutation(seed=20260929)[1000:1256], disjoint from the ENTIRE previous 1000-query development pool, not only the 128 calibration queries. The implemented experiment differs from an earlier proposed package: it evaluates one >=0.99 recall target on 256 held-out queries, not two targets on 512 queries.

IVF coarse centroids are shared, nlist=1024. The development nprobe grid is 8/16/32/64/96/128/192/256 for the in-memory comparison, and through 128 for the disk GeoIVF arm. Each method selects its fastest eligible development setting before held-out evaluation. All IVF arms select nprobe=64. DiskANN searches L=20/40/60/80/120/200/400 at beams 4/8 and selects L=60, beam=8. No held-out retuning is performed.

GeoIVF uses the previously selected PCA64 8-bit balls, prepared planner, SIMD exact scan, 256-page windows and gap2 coalescing. Original GeoPack payload SHA256 remains 8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7. The learning-file PCA fingerprint remains d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64.

## 1. Held-out in-memory execution

Five randomized interleaved rounds, 256 distinct queries, 1280 measurements per method. Full vectors reside in memory for all arms in THIS table, including GeoIVF MemoryReplay. Do not label GeoIVF replay as a 74.6 MiB deployed system: its replay payload is additionally resident.

| Method | nprobe | Mean ms | Median ms | p95 ms | Strict Recall@10 |
|---|---:|---:|---:|---:|---:|
| Packaged Faiss 1.15.1 IVFFlat | 64 | 0.886567 | 0.876846 | 1.039125 | 99.53125% |
| Released HIVF-CLIP | 64 | 0.900909 | 0.961416 | 1.181903 | 99.453125% |
| CLIP-bundled Faiss IVFFlat | 64 | 1.148526 | 1.135722 | 1.383606 | 99.53125% |
| Released IVF-CLIP | 64 | 1.152423 | 1.233312 | 1.570517 | 99.53125% |
| GeoIVF RAM replay | 64 | 2.344631 | 2.318737 | 3.298494 | 99.53125% |

GeoIVF replay is 2.6446 times slower than packaged Faiss here. No in-memory speed win is established. The preceding internal IVF speedups did not compare against these released implementations.

IVF-CLIP is approximately tied with its bundled Faiss at this point; it is not uniformly faster at every nprobe. On DEVELOPMENT queries, HIVF-CLIP at nprobe=128 takes 1.433946 ms versus 1.725393 ms for packaged Faiss, at recalls 99.765625% and 99.84375% respectively. That is a development operating point, not a new held-out claim or exactly matched recall. Do not infer a universal ranking from one selected nprobe.

Fixed-nprobe oracle checks: packaged Faiss, bundled Faiss, IVF-CLIP and GeoIVF each match the independent exhaustive IVF-candidate-set answer in all 256 queries. HIVF-CLIP matches 255/256. These are observed checks, not a universal no-loss guarantee for learned CLIP pruning. The raw same-nprobe CSV records equality/recall, not every neighbor array.

## 2. Held-out disk-provider execution

Three rounds on the same 256 held-out queries. Both configurations were selected for >=99% recall on DEVELOPMENT. Held-out recall is not numerically identical, and the implementations currently report DIFFERENT recall conventions: GeoIVF uses strict supplied top-10 IDs, DiskANN uses upstream tie-aware recall. This is an equal-target pilot, not an exactly equal-recall final leaderboard.

| Method | Mean query ms | Tail | Reported held-out recall |
|---|---:|---|---:|
| Released DiskANN3 disk provider, L=60, beam8 | 1.610077 | per-round p95 1.782 to 1.834 ms | 99.375%, upstream tie-aware |
| GeoIVF, nprobe64, window256, gap2 | 6.419944 | pooled median 6.179139 ms; p95 11.179493 ms | 99.53125%, strict ID recall |

DiskANN round means are 1.628363, 1.608930 and 1.592938 ms. The descriptive ratio of mean latencies is 3.9874 in DiskANN's favor. Do not advertise this as a rigorously exact-recall-matched speedup.

DiskANN reports microseconds in its JSON. Its outer search-loop spans independently imply 1.611270 ms/query averaged over the three rounds, close to the 1.610077 ms internal statistic. The broad difference is therefore not explained by simply confusing milliseconds and microseconds. Nevertheless, DiskANN library timing and GeoIVF full Python/application timing are not identical scopes; loading/building the DiskANN index is excluded from service latency.

DiskANN node caching is explicitly disabled (num_nodes_to_cache=null, upstream None), and all held-out cache-hit counters are 0%. PQ codes remain RAM-resident; node-cache disabling does NOT mean an empty CPU, OS or device cache. Disk build: max_degree=64, l_build=100, FP32 disk vectors, 64 PQ chunks, four build threads. Held-out searches use one thread and inherit the same CPU-0 affinity as GeoIVF. GeoIVF performs actual pooled io_uring/O_DIRECT reads at queue depth16. Upstream search/training code is not altered by the harness.

## 3. Memory and stored-data tradeoffs

These are COMPONENT accounts, not matched whole-process resident-memory caps.

| Component | Size MiB | Interpretation |
|---|---:|---|
| Faiss FP32 vectors only | 488.28125 | calculated 1M*128*4 |
| Faiss vectors + 64-bit IDs + coarse centroids | 496.410645 | calculated core arrays; allocator/list/runtime overhead extra |
| IVF-CLIP per-vector radial scalar | 3.814697 | additional to full vectors/IDs; tables/runtime also extra |
| GeoIVF complete directory arrays | 74.576189 | recorded 78,198,802 bytes; excludes runtime and per-query I/O buffers |
| DiskANN PQ code file | 61.035164 | 64,000,008 serialized bytes; not total runtime RAM |
| DiskANN PQ pivots file | 0.129665 | 135,964 serialized bytes; not a complete runtime allocation model |
| GeoIVF vector backing file | 490.019531 | 513,822,720 bytes including fixed-page padding |
| DiskANN graph-plus-vector backing file | 781.253906 | 819,204,096 bytes; PQ files are separate |

GeoIVF's directory is about 15.0% of the calculated Faiss vectors/IDs/centroids core. This supports the intended compact-directory-over-disk architecture, not RAM-like speed. It is NOT the live memory footprint of the RAM-replay arm, which additionally loads its payload.

The recorded DiskANN PQ component is smaller than GeoIVF's directory; these different accounting scopes cannot establish total-RAM dominance. They do show why 'DiskANN wins because it holds all original vectors in RAM' is not supported by this setup. Per-worker scratch, graph metadata, centroids, allocators and transient I/O buffers must be measured consistently in separate deployment processes before claiming an equal-memory frontier.

GeoIVF's vector backing file is 37.2778% smaller than DiskANN's graph-plus-vector file in this build. This is a concrete storage-density difference, NOT a comparison of all persisted assets or a general advantage over every DiskANN graph/layout configuration.

## 4. Work and I/O

On held-out GeoIVF queries the means are:
- 8387.527344 candidate pages, 326.816406 selected pages, 53.906250 bridged-gap pages.
- 380.722656 pages actually requested after bridging, or 1,559,440 bytes (1.487198 MiB) per query.
- 99.082031 coalesced requests and 26.378906 dependent read stages.
- 3039.941406 full-vector distance evaluations.

Thus candidate-page elimination remains 95.4608% on held-out queries. Strong pruning survived the development/test split; it is not sufficient to make this implementation faster than the released disk graph baseline.

DiskANN reports 85.4375 mean I/O operations and 3541.070313 comparisons. Those comparison counters include different kinds of distance work than GeoIVF's exact-scan counter. Likewise, its I/O counter comes from an upstream tracker; this analysis has NOT normalized it to GeoIVF's coalesced requests, 4 KiB pages, bytes, distinct sectors or physical NVMe commands. Do not multiply it by 4096 and call the result measured storage bytes without auditing the reader/counter semantics.

GeoIVF's direct application categories average 1.676514 ms filtering/coalescing, 4.130120 ms read stages, and 0.436698 ms exact scan/top-k, plus other coordination. DiskANN's reported internal averages are approximately 0.134464 ms CPU, 1.473582 ms I/O and 0.002031 ms PQ preprocessing. These timing categories have different instrumentation boundaries; use them as diagnostic pointers, not a precise additive cross-engine Amdahl proof.

Inference: a page filter on a broad IVF candidate set and graph-guided candidate discovery perform different amounts and kinds of work. The opportunity is not merely another byte saved per ball. Fewer examined summaries, better candidate-level pruning and fewer dependent requests may be needed. A filter that preserves a prescribed IVF candidate answer also has a different contract from a new ANN graph, but that contract alone is not a speed advantage.

## 5. Build, correctness and experimental limits

The archived IVF build components are 1.402 s for bundled Faiss population with precomputed coarse centroids, 30.807 s for IVF-CLIP and 95.366 s for HIVF-CLIP (including calibration/hierarchy work). DiskANN reports a 60.847 s graph/PQ build using four threads. GeoIVF construction is not separately broken out here. These have DIFFERENT included steps and thread budgets and must not be ranked as a normalized construction-cost comparison.

Both campaigns ran the existing 260-test suite: the disk campaign passed all 260 with no skips; the in-memory campaign passed 242 with 18 optional backend tests skipped. These are regression suites, not 260 new tests of the external measurement bridge.

This analysis ran 254 independent artifact-consistency assertions: archive SHA256s, common query splits, complete CSV rows/repeats, aggregate means/medians/p95, frozen development choices, GeoIVF repeatable counts, byte/page and selected/gap identities, memory component formulas, per-query recall agreement between Geo replay/direct, raw upstream outputs versus summaries and test XML counts. This is analysis of recorded evidence, NOT a rerun of every neighbor search. DiskANN exports only aggregate results here, so its individual result IDs and per-query latency distribution cannot be independently reconstructed from these artifacts.

The experiments are a single dataset and construction seed on a shared AMD Ryzen 9 9955HX3D host. A file lock serializes GeoIVF benchmark campaigns, not every process on the laptop. CPU frequency and device-cache state are uncontrolled; payloads fit host RAM and oracle/base arrays coexist in the Geo benchmark process. No equal RSS cap, larger-than-RAM scale result, concurrent throughput test, or instrumented physical-I/O comparison exists. DiskANN development calibration precedes affinity pinning; its held-out runs are pinned. Disk arms alternate in blocks across rounds rather than query-by-query interleaving. In-memory methods are query-interleaved.

## Decision

The internal IVF acceleration remains real, and the 95.46% held-out page reduction is encouraging. The external evidence does NOT establish a SOTA speed win: GeoIVF replay is slower than packaged Faiss, and its disk-backed implementation is slower than the tested released DiskANN provider at a common >=99% target. The apparent storage-density advantage and transparent-IVF pruning contract deserve separate evaluation; neither should conceal the latency result.

The next controlled comparison should place IVF-CLIP and IVF-CLIP plus GeoIVF's page directory on the SAME disk backend and fixed total-RAM budget, while harmonizing DiskANN result/recall and memory accounting as an external reference. Keep earlier line/box variants as ablations. Do not choose new parameters from these 256 held-out queries and continue describing them as untouched test data.

Reproducibility artifacts: external-ivf-results.json, external-disk-results.json, frozen-selection.json, development/heldout CSVs, DiskANN raw input/output JSON, trained CLIP tables, source.zip and test logs in the two run archives. Retain the archives beyond GitHub's 30-day artifact retention period.
