# NavHints complete-request latency study: Coveo, Text2Image-10M, BigANN-10M

Status: cross-workload harness implemented and static checks passed; native measurements
must complete and be audited before publication. PubMed's separately validated
12-width, 3-rotation, 3-Catapult-seed benchmark is GitHub run
[38259064842](https://github.com/gilga1983/GeoIVF/actions/runs/38259064842),
artifact 11756292555.

This extension measures the **three remaining paper workloads**, using the
identical pinned Microsoft DiskANN graph/SSD for all arms *within each workload*.
It reuses each dataset's existing frozen SSD index, PQ representation,
trained 512-region junction selector, disjoint static history and causal
query replay. No change to the graph, selected `B<=14464,C=2048,k=8`
RAM-only Core, or baseline semantics.

## Study design

| Dataset | Vectors, metric | Static training | Online warm-up | Measured suffix |
| --- | --- | --- | --- | --- |
| Coveo eCom | 31,950, float32 inner product | 5K | 20K | 5K |
| PubMed/MedCPT (already complete) | 1M, float32 inner product | 5K | 4K | 1K |
| Text2Image-10M | 10M, float32 inner product | 5K | 4K | 1K |
| BigANN-10M | 10M, uint8 squared-L2 | 5K | 4K | 1K |

Arms: native-medoid DiskANN, junction-only routing, NavHints Core, and
Catapult's source-aligned hash/LRU destination routing adapted onto the same
DiskANN SSD backend. Catapult's original C++/author engine is not being
benchmarked; it is explicitly a policy comparison. Each Catapult seed observes
the first 5K queries to create a separate state snapshot for each L, then
receives the same held-out causal sequence as every other method. Training's
ground-truth file is intentionally zero-filled and `DISKANN_SKIP_RECALL=1`:
it cannot be confused with an oracle answer cache.

All arms run one in-flight query (four internal search threads, beam 8,
Recall@10), so history updates from completed queries only. The baseline and
junction-only arms execute **the same causal outer loop**, including the
20K/4K warm-up, instead of being evaluated from a different buffer-cache
condition. Catapult bucket capacity is adjusted so its **accounted routing
payload** is at least as large as NavHints' payload for the dataset's
dimension; no complete RSS equality is claimed.

Run the 2-width single-seed **smoke** first. A full workload candidate is
6 widths `L=10,20,40,80,160,320`, three rotated executions and three
Catapult hash seeds. The tested search widths differ from PubMed's 12-point
grid but all per-dataset comparisons are matched recall and never
extrapolate beyond actual supported overlap. Running all three datasets can
take substantial disk time; workflow matrix is sequential and uses
`$HOME/.cache/geoivf/speed-device.lock` for compatibility with the other
experiments. No new datasets are downloaded by this script.

## Recorded evidence and interpretation

The outer service timer includes routing, in-memory PQ scoring, disk-index
search, result-copy bookkeeping and post-query FIFO/LRU updates. Its
`service_us` is the **complete per-query wall clock** for the causal
replay. Native `engine_us` and `io_us` are diagnostic, do not sum them
because they can overlap. Native reads are **logical provider SSD requests**.

- p50, p95 and p99 are computed from **directly measured per-L raw requests**.
  Do not interpolate percentiles across L and label them matched recall.
- Mean complete-service latency and SSD read cost are compared at common
  Recall@10 by within-support interpolation of the **absolute per-run**
  recall--cost curves. The report retains both three-arm and four-arm
  valid target subsets. A missing Catapult overlap must never silently
  suppress the three-arm result or be extrapolated.
- Run-to-run differences are descriptive spread on the same frozen workload,
  not a confidence interval for new independent query populations.
- Cross-workload percentages must not be pooled into a universal latency
  reduction or compared without noting different matched-recall intervals.
- Keep negative results. BigANN/Starling previously showed that additional
  candidate scoring may cost CPU without sufficient extra I/O savings.

## Code, actions and reproduction

- [Implementation](../scripts/qualify_navhints_service_multidata.py)
- [Frozen runner](../scripts/run_navhints_service_multidata_ci.sh)
- [Self-hosted Action](../.github/workflows/navhints-service-multidata.yml)
- [Regression tests](../tests/test_navhints_service_multidata.py)

The completion report writes
`results/complete-service-{dataset}.json`, per-query CSVs and native
input/output JSON, logs, Catapult training-snapshot hashes, source patch
diffs, input/index/IVF SHA256, and environment details. This new latency
study is separate from existing frozen logical-SSD-read experiments;
there is no reason to rerun prior competitors merely to update their
previous table values.

The manuscript subsection and plots must be produced from independently
validated result JSONs, not from early benchmark stdout or per-run
speculation.
