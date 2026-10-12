# NavHints complete service latency experiment (2026-10-12)

Status: instrumented candidate; do **not** cite numbers in the manuscript until a
successful native run and result audit. Selected NavHints Core is frozen.

## Why this experiment exists

The paper's previously reported `mean_latency` and `qps` fields are not
comparable complete-service metrics across author-aligned Catapult and the
NavHints causal replay. In particular, NavHints learns after the native search
timer stops, and Catapult's native timer similarly excludes its online LRU
feedback. This experiment adds an outer per-request monotonic timer to the
**same pinned Microsoft DiskANN SSD engine** for all four arms.

## Frozen protocol

- Graph/engine: DiskANN commit `fcf90534174cf29c78c9f13b4cccf1fcabff85f5`,
  one frozen PubMed/MedCPT 1M SSD index and matching resident PQ codes.
- Exact same frozen 10K query sequence: 5K disjoint training, 4K chronological
  online warmup, 1K chronological measured requests. Causal history updates
  occur only after the prior request completes.
- Search parameters: Recall@10, beam width 8, four internal search threads.
  Exactly **one in-flight query** to retain strict causal feedback order.
- Four arms, rotated within each L: native medoid baseline, learned junction
  selection only, selected RAM-only NavHints Core with (B,C,k)=(14464,2048,8),
  and source-aligned Catapult hash-bucket LRU routing, learned from the disjoint
  static prefix and updated during held-out replay.
- Catapult is a policy port to this common SSD search engine, **not** the
  complete native author's in-memory Catapult system.
- The 102,940-byte NavHints memory budget is *accounted routing payload*;
  this experiment does **not** establish equal RSS.
- Shared `$HOME/.cache/geoivf/speed-device.lock` serializes source builds and
  all SSD replays against other NavHints workloads. Artifacts include binary,
  inputs and navigation-directory SHA-256 digests, exact source patch diffs,
  host details, snapshots, raw query timings and aggregation.

## Measurements and interpretation

The outer timer starts immediately before query search and ends when query
completion/learning ends. The per-query CSV contains the measured suffix only:

- `service_us`: actual request wall time, including routing, graph navigation,
  SSD read and post-query feedback.
- `engine_us`: the adapter's existing per-query search timer (which excludes
  feedback). This is a *subset* of the outer interval.
- `remainder_us`: `service_us - engine_us`; includes result copying and Rust
  bookkeeping **as well as** feedback. Do not mislabel as feedback alone.
- `io_us`: native provider I/O timer; it may overlap other work and should
  **not** be added to `engine_us`.
- `pq_preprocess_us`: native PQ preprocessing; may already be included inside
  other timers and is **not** by itself total routing scoring.
- `reads`: logical SSD-provider graph record reads, not physical NVMe commands.

CSV data are summarized as p50/p95/p99 complete-request latency, mean service
time, serial throughput (1000 / sum of measured service times), means of
native I/O and CPU counters and all 1000 exact query identities/order positions.
High percentiles are computed from per-request samples; the same frozen stream
is replayed in repetitions, so no population-confidence intervals may be
derived from the rotations alone.

At equal Recall@10, interpolate means of absolute cost curves **inside** the
supported overlap only, without extrapolation. High percentile comparisons
remain direct per-L diagnostics unless operating points match recall.

## Execution tiers

The GitHub Action has two tiers. Initial smoke (L=40,80,160, one repetition,
one Catapult seed) validates instrumentation and serial causal behavior;
it is **not a publishable full performance comparison**. Full tier (12 widths,
three rotated repetitions, three Catapult seeds) is the manuscript candidate,
subject to consistency checks and SSD noise review.

Workflow:
`.github/workflows/navhints-service-latency.yml`

Or on the authorized self-hosted Legion runner:
`bash scripts/run_navhints_service_latency_ci.sh`
with `SERVICE_WIDTHS`, `SERVICE_REPS` and `SERVICE_CAT_SEEDS` set.

The workflow does not modify or regenerate the paper. Once the full curves
exist, add a separate plot/table with validated source SHA and exact artifact
identifiers. Increasing query concurrency requires a distinct experiment and
an explicit policy for concurrent feedback; do not claim multi-client QPS from
the serial run.
