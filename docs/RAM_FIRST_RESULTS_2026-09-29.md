# RAM-first verification results, 2026-09-29

## Provenance and protocol

Tested implementation: `3471eb7ab3bc8e03fd52ccb1ef923c045aeab06a`.
Successful workflow: `36616097765`; job `109569366716`.
Artifact `11055049002`, `ramfirst-36616097765`.
Downloaded ZIP SHA256:
`41d622b5bc71bcc7b09cf2b9f62d078e6409eb95d180c054e5897dbc109524ea`.

The full canonical SIFT1M database is unchanged: one million original 128D FP32
vectors, 4 KiB pages, eight vectors/page, 125445 pages, nlist=1024, nprobe=64,
k=10 and construction seed12345. Reuse frozen GeoPack pages, learning-file PCA64,
eight-bit coordinate codes and their measured-error radii. No payload compression,
new layout, external candidate index or newly learned ranking model is introduced.

The native ranker scans compressed entries in selected IVF lists and maintains a
shortlist by projected-center distance. It consults no original vectors during
ranking. Fetch distinct shortlisted pages and evaluate all their original vectors.
Approximate mode stops after this first verification wave. Certified mode also
computes conservative page bounds during the RAM scan; it fetches every unread
page not excluded by the first wave's exact kth threshold as a second wave.
Quantization radii remain in the common live directory for both modes. They are
unused in approximate ranking, but no radius-memory saving is claimed here.

RAM-first verification has no gap bridging. The control is the previously selected
staged implementation, with prepared selection, SIMD exact scan, window256/gap2.
Both use the same pooled rolling io_uring/O_DIRECT reader at queue depth16 and the
same original-vector FP64/SIMD top-k kernel. RAM-first waves may be split into
bounded batches without recomputing their admission threshold. In this campaign,
each verification wave fit a single reader call. This is not a claim of one or
two SSD commands: a wave contains many extent requests.

Development uses 128 queries from permutation(seed=20260929)[:128]. Sweep R in
16/32/64/128/256 for approximate and certified modes. Three randomized interleaved
direct-I/O rounds choose the fastest member of each family meeting 99% strict
Recall@10. Choices are frozen before timing fresh test queries. This fixed-nprobe
experiment does not jointly tune nprobe, shortlist and memory.

The NEW held-out cohort is permutation[1256:1512], 256 distinct queries. It excludes
the entire original 1000-query development pool and the 256 previously examined
external-baseline test queries. Only staged search and the two frozen choices are
evaluated on these new queries, with three rounds each of direct I/O and RAM replay.
All parameter selection is development-only. No test-set retuning occurs.

## Development sweep

Means over 128 development queries and three timing rounds. This table is for
configuration selection, not held-out evidence. Geometry and candidate routes are
identical across rows; higher shortlist sizes can add approximation accuracy but
also add page reads. The reference IVF recall on this cohort is 99.375%.

| Mode | R | Mean ms | Strict Recall@10 | Pages/query | Verification waves/query |
|---|---:|---:|---:|---:|---:|
| Staged control | - | 10.6632 | 99.3750% | 393.4063 | 26.1250 |
| Approximate | 16 | 4.5690 | 90.5469% | 14.5938 | 1 |
| Approximate | 32 | 4.7100 | 98.6719% | 28.4297 | 1 |
| Approximate | 64 | 5.1016 | 99.3750% | 54.7969 | 1 |
| Approximate | 128 | 5.6184 | 99.3750% | 104.8125 | 1 |
| Approximate | 256 | 6.6424 | 99.3750% | 198.1016 | 1 |
| Certified | 16 | 8.8123 | 99.3750% | 196.4688 | 2.0000 |
| Certified | 32 | 8.5603 | 99.3750% | 176.7500 | 1.9531 |
| Certified | 64 | 8.6220 | 99.3750% | 177.3594 | 1.8750 |
| Certified | 128 | 8.5941 | 99.3750% | 187.7422 | 1.6094 |
| Certified | 256 | 8.9719 | 99.3750% | 238.3438 | 1.2969 |

Frozen choices: approximate R=64 and certified R=32. Certified sizes32/64/128
have closely spaced development means, so do not claim R=32 is a robust universal
optimum. A sufficiently large shortlist can produce the correct answer without
certifying it: approximate R64/128/256 happened to match the candidate oracle on
all development queries, while certification still often required more reads.

## Fresh held-out direct-I/O result

Complete online-query wall time includes fresh routing, projection, compressed
ranking/filtering, coalescing, actual direct reads, exact scan and top-k. Each row
has 768 timed measurements but only 256 DISTINCT test queries.

| Method | Mean ms | Median ms | p95 ms | Strict Recall@10 | Pages/query | Read calls/query |
|---|---:|---:|---:|---:|---:|---:|
| Staged control | 10.6437 | 10.2969 | 19.9440 | 99.5703125% | 394.0313 | 25.3867 |
| Approximate RAM-first, R64 | 5.1772 | 5.1443 | 5.9083 | 99.4921875% | 55.2578 | 1.0000 |
| Certified RAM-first, R32 | 8.8014 | 8.3436 | 12.6688 | 99.5703125% | 171.9727 | 1.9648 |

Approximate RAM-first reduces mean latency 51.3589%, a 2.0559x same-campaign
speedup, and requests 85.9763% fewer pages. Mean requested traffic is 226336 bytes,
about 221.03 KiB/query, through 47.5117 extents. It matches all ten IVF reference
IDs on 254/256 queries. Each of the other two queries misses ONE IVF-reference
neighbor. IVF-reference Recall@10 is 99.921875%; global strict recall drops by
0.078125 percentage points. This is a measured approximation, not zero loss.

Certified RAM-first reduces mean latency 17.3087%, a 1.2093x speedup, and requests
56.3556% fewer pages while matching the candidate-set oracle on all test queries.
Its first wave reads 28.6992 pages/query; its completion wave adds 143.2734.
Nine of 256 queries certify after the first wave; the other247 need the second.
Mean total requested traffic is 704400 bytes through 142.7695 extents. It issues
MORE extents than staged search (97.5117), but in far fewer dependent waves.
Its speed gain combines reduced bytes and changed dependencies; this is not an
isolated causal measurement of wave count alone.

Per-round mean latencies, staged / approximate / certified:
- Round0: 10.82928 / 5.15660 / 8.85534 ms.
- Round1: 10.62586 / 5.19028 / 8.83471 ms.
- Round2: 10.47593 / 5.18476 / 8.71418 ms.

The price of certification cannot be obtained solely by subtracting the two
selected held-out means because they use DIFFERENT shortlist sizes. Same-R
comparisons are available in the development table. At R64, for example, the
certified method performs extra RAM bound work and second-wave reads relative
to the approximate arm. Do not describe the certificate as free.

## CPU work: the remaining limitation

| Method | RAM-replay mean ms | Direct rank/filter/coalescing ms | Direct read-stage ms | Direct scan/top-k ms |
|---|---:|---:|---:|---:|
| Staged control | 3.6697 | 2.6563 | 6.7025 | 0.9119 |
| Approximate R64 | 4.2399 | 3.8585 | 0.9490 | 0.1711 |
| Certified R32 | 6.5649 | 5.7843 | 2.2547 | 0.4856 |

RAM-first is NOT faster in this memory-replay comparison. The native ranker
currently evaluates all64 coordinates for every selected-list proxy and maintains
a shortlist. Staged rejection can stop after partial distances. Approximate
ranking alone costs 3.7239 ms/query in the direct campaign; certified ranking,
including bound construction and radial combination, costs 4.5659 ms. Certified
unread-page filtering adds another0.8335 ms, before coalescing. These timer scopes
are explicit implementation categories, not a proof of an optimal CPU cost.

The win is less storage waiting, bought with more RAM-side work. Approximate mode
reduces complete-query process CPU from 5.8305 to 4.4814 ms because it also avoids
most exact scans/wrappers. Certified mode increases it to 7.2625 ms while still
reducing wall latency. Memory bandwidth/CPU optimization of ranking and certificate
bookkeeping are now direct follow-up targets; no such optimization is claimed in
this first architectural test.

## Correctness, memory and independent artifact verification

All283 runner tests passed, with no errors, failures or skips, including23 new
RAM-first tests and existing actual O_DIRECT/io_uring regressions. The local suite
passed264 with19 optional dependency skips. Local selection-contract checks also
verified frozen-choice behavior when a recall target is infeasible.

The initial1408 configuration-query checks cover11 methods on128 development
queries. All staged/certified answers match the independent exhaustive FP64 oracle;
approximate differences are measured rather than asserted away. There are8832
timed searches in total:4224 development-direct,2304 held-out-direct,2304 held-out
memory. All required exact answers and all repeated method/query answers and
operation counts agree.104474 rejected-page decisions were audited after search;
all passed. Audits cover the first two development queries in each certified arm
and the staged arm and repeat pages; they are not independent page samples.

The existing summary builder certifies coverage of every indexed point. Tests
also verify unique page reads, exact top-k over fetched page contents, cross-list
coalescing boundaries, already-fetched gaps, bounded batch splitting and identical
ranked shortlists between approximate and certified modes. Certified correctness
rests on conservative bounds and full completion, not the shortlist's recall.
Numeric safeguards are engineering measures, not formal floating-point proofs.

After download, the artifact SHA matches GitHub. All six new executable/workflow
files in source.zip match the locally tested copies byte-for-byte. Independently
reconciled8832 CSV rows,2944 saved per-query neighbor arrays,17 aggregate rows,
all byte/page and wave-count equations, frozen choices, query disjointness,
strict/global reference recall, and test XML counts. These checks analyze the
recorded artifacts; they do not rerun the full vector corpus locally.

Persistent directory arrays remain78198802 bytes (74.576 MiB), excluding runtime
and query scratch. The campaign's shared I/O pool peaked at23171072 bytes
(22.098 MiB), under its64 MiB cap; this includes idle reservations across all
shortlist variants and is not a per-arm steady-state minimum. Query scratch adds
candidate-page arrays/bounds, shortlisted slots/scores and visited-page sets.
The approximate mode still loads the common radii array; metadata savings from
omitting it were not measured. Campaign RSS includes base/oracle/scratch data and
must not be described as deployed index memory.

## Scope and decision

One SIFT1M index and construction seed; shared AMD Ryzen host with CPU0 affinity,
not an exclusive core. Host load averages were15.6875/15.3550/14.7129 at timing
start and16.0659/15.4946/14.8032 at end, materially busier than previous campaigns.
The staged baseline is therefore rerun alongside all new arms; compare within
this campaign, not the earlier approximately6ms figure. CPU frequency/device
caches are uncontrolled. The payload fits host RAM and the benchmark process
retains base/oracle arrays, although timed direct reads really use O_DIRECT.

No fresh DiskANN, CLIP or upstream-Faiss run is included. These timings are not
a SOTA win or a matched ratio against earlier external numbers. Concurrency,
large-scale residency caps and independently instrumented physical I/O remain
unmeasured. Future parameter tuning must not call these256 test queries untouched.

The architecture is promising: one-wave approximate verification halves latency
with a small measured additional recall loss; two-wave certification preserves the
IVF answer and improves latency modestly, with much lower requested traffic. Both
are retained as opt-in modes. Defaults remain unchanged. Optimize the RAM ranker
and certificate bookkeeping before making broader external-speed claims.
Reproduction: docs/RAM_FIRST.md and scripts/qualify_ramfirst.py.
