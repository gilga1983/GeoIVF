# Final NavHints Core experiment inventory (frozen evidence)

The ten primary/sensitivity campaigns below **completed successfully on a self-hosted real-SSD runner**, except where an individual workflow has more than one independently successful job. This is the **original experimental history**, including the optional
Sample2 persistent-exit arm. The submitted paper's headline algorithm is
RAM-only Core. The currently active per-figure/run mapping is in
[ARTIFACT_MAP.md](ARTIFACT_MAP.md); the older Full-anchored Sample2 percentages
below must not be substituted for the RAM-only paper results.

The exact run IDs, artifact names, source branch and member paths are machine-readable in [repro/manifest.json](../repro/manifest.json). They remain the authoritative original cohorts even if new GitHub branches later receive experimental commits.

| ID | Successful GitHub Actions run | Artifact | Reproduces |
| --- | --- | --- | --- |
| coveo | [38136319610](https://github.com/gilga1983/GeoIVF/actions/runs/38136319610) | 11701308908 | Production-order 5K/20K/5K, four arms, final Core/Full |
| pubmed | [38137049239](https://github.com/gilga1983/GeoIVF/actions/runs/38137049239) | 11702879389 | PubMed1M final Core and Sample2 vs DiskANN |
| text2image | [38136365199](https://github.com/gilga1983/GeoIVF/actions/runs/38136365199) | 11702522173 | Text2Image-10M (independently successful matrix job) |
| bigann | [38136365199](https://github.com/gilga1983/GeoIVF/actions/runs/38136365199) | 11704362848 | BigANN-10M (independently successful matrix job) |
| catapult | [38127358063](https://github.com/gilga1983/GeoIVF/actions/runs/38127358063) | 11700450279 | Original-author-aligned Catapult hash/LRU policy vs selected RAM-only Core |
| memory | [38120556463](https://github.com/gilga1983/GeoIVF/actions/runs/38120556463) | 11696380166 | Matched-payload B/C/k tradeoff on PubMed |
| starling-coveo | [38137155521](https://github.com/gilga1983/GeoIVF/actions/runs/38137155521) | 11702922642 | Selected Core on original native Starling, Coveo |
| starling-bigann | [38137155521](https://github.com/gilga1983/GeoIVF/actions/runs/38137155521) | 11705972486 | Selected Core on original native Starling, BigANN |
| sensitivity | [38137378869](https://github.com/gilga1983/GeoIVF/actions/runs/38137378869) | 11707480968 | Selected-Core warmup, controlled locality and train-support shift |
| persistence | [38137546048](https://github.com/gilga1983/GeoIVF/actions/runs/38137546048) | 11703989015 | Selected-Core Sample1/2/4/10 + no-seeding, reads and logical write requests |

## Four-workload matched-recall headline data

These are the submitted paper's RAM-only Core results, with
`B<=14464,C=2048,k=8`. Values average per-target savings over the common
recall support; workload-specific target grids are not pooled.

| Workload | Core read saving vs DiskANN | Targets | Complete-request latency saving |
| --- | ---: | ---: | ---: |
| Coveo | 80.15% | 4 | 68.32% |
| PubMed1M | 44.13% | 5 | 28.85% |
| Text2Image-10M | 34.57% | 5 | 20.46% |
| BigANN-10M | 21.03% | 4 | 16.08% |

Latency has a separately documented matched-recall grid and includes routing
and feedback in the common DiskANN engine. The optional Sample2 persistent
extension has separate Full-anchored summaries in the frozen evidence; it does
not define the paper's headline RAM-only results.

The four-workload service-latency runs and all newer figure/table sources
are mapped in [ARTIFACT_MAP.md](ARTIFACT_MAP.md), including the recovered
Table 5 destination controls. Use `make review` for numerical verification.

## Native original Starling portability

Both native-Starling jobs completed successfully at the original author commit `17dc3e8a011533a62374445f53963e951b72883a`. The original Starling memory navigator and disk layout remain enabled. All arms use the same native index and causal query order.

- **Coveo:** final Core saves an equal-weight **41.75% native logical I/O** versus original native Starling and about **28.60% matched-recall query latency**, across the three mutually reachable baseline anchors. Strong demand concentration supports worthwhile extra route choices.
- **BigANN-10M:** final Core's improvements are **not consistently positive**. Its native baseline-matched savings over four anchors are approximately `-0.006%, -0.537%, +0.315%, +0.202%`, for a mean near **zero**. This is evidence that an already optimized in-memory navigator can make additional routing work marginal or harmful when demand locality is weak.

Keep native Starling `n_ios` counts separate from physical NVMe operations and do **not** divide measured QPS between different engines.

## Published Catapult and memory conclusions

The author-aligned Catapult routing policy uses original Gaussian hyperplane/hash packing, per-bucket LRU of winners and request-causal updates on the **same** pinned DiskANN SSD engine. Across 11 mutually reachable PubMed Recall@10 anchors, the selected RAM-only Core saves **13.28% logical SSD reads**, including **3.08% at 79.81% recall**. This is a routing-policy comparison, not original author's whole in-memory Catapult system.

For the same **102,940-byte accounted routing payload** on PubMed, the final `B14464/C2048/k8` allocation saves **7.42% matched-recall logical reads** relative to `B16000/C512/k1` across eight common anchors. Because the Catapult study uses a different eleven-anchor grid, it reports **8.49%** versus that compact-history control. Both are valid descriptive summaries of **different** recall grids, not results to average together.

## Sampled persistence

On PubMed, all policies use the same final RAM-only Core and optional H=10 exit slots. Across four common matched-recall anchors the RAM-only Core saves **19.41%** relative to learned junctions; Sample2 saves **19.68%**, only **0.33% additional reads versus Core**. Sample2 performs **0.264 logical junction-page update requests per measured query at fixed L=160**. Sample1/2/4/10 are alternative admission controls, not separate Core architectures. These write counts are **not** evidence of physical flash amplification or crash recovery.

## Data/claim boundaries

- Disjoint static training and causally updated online state; no held-out GT used as search hints.
- Three rotated replays of **the same** deterministic query suffix, not three independent workload draws.
- All storage-read results from actual pinned SSD engines; match on **Recall@10**, not raw search width.
- Payload 102,940 bytes is **accounted ID/group/PQ-representative storage**, not matched total process RSS.
- CatapultDB is source-aligned **routing** in DiskANN; native Starling is the actual author engine.
- Negative BigANN/Starling results remain first-class measurements in paper discussion.
- No user-identifying or licensed Coveo data are distributed through the repository.

The pinned plotting sources and derivations are bundled under `repro/paper/`. `make paper-data` regenerates and checks the plotted CSVs without the private manuscript repository.
