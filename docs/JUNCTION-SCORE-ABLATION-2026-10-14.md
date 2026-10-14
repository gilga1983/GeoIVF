# NavHints junction-scoring ablation (research question RQ2)

**Status:** Await frozen real-SSD measurements from [GitHub Actions run 38525797031](https://github.com/gilga1983/GeoIVF/actions/runs/38525797031). This is exploratory evidence and has **not** been introduced into the paper.

## Question

The original score `S(v) = sum_{q expanding v} p_q(v)` combines
frequency with the depth at which a vertex is first expanded. Does that
actually outperform (a) visit frequency alone and (b) average first
expansion position on a matched-memory basis?

## Controlled experimental design

- Compare **sum position** (approved/current), **count** (unique
  training traversals visiting vertex), and **mean position** (sum/count).
  Exclude initial starting vertices and deduplicate repeated IDs per query.
- Train each ranking from the **same exact 5,000-query, frozen teacher
  trace**. Each retains exactly **B=14,464** distinct junction IDs.
- Generate each directory with the original dataset's **unchanged**
  512-region routing procedure and seed, with identical nprobe=8.
  Recluster the newly selected IDs using that same procedure: region
  memberships and medoids naturally depend on which IDs the score retains.
  Do not mischaracterize these experiments as identical geometric medoids.
- Require the `sum` variant's ID file **and its full regional directory**
  to match the preexisting selected-Core files byte-for-byte.
- Preserve the original Microsoft DiskANN graph, training/test split,
  PQ representation, C=2,048 unique-FIFO destination history, k=8
  extra starts, and zero persistent-hint writes for all variants.
- PubMed/MedCPT (inner product) and BigANN-10M (squared L2) use the
  original real-NVMe storage indexes and disjoint 5K train, 4K causal
  warm-up and 1K measured replay. Sweep identical L=10/20/40/80/160/320.
- Run three repetitions, rotating score-variant execution order.
  Calculate *matched-recall logical SSD reads* anchored at reference
  `sum`-score measured recall points jointly reachable by all variants.
  Never extrapolate; require at least three jointly supported points.
- Compare source-score ID overlap and per-recall results. Count repeated
  executions as technical replays of the same fixed query population,
  not independent workload samples. Avoid end-to-end latency claims
  from cross-run timing without controlled measurement.
- **No external competitors need rerunning**. This isolates a choice
  *inside* NavHints, not a new baseline bake-off.

## Reproduction

`scripts/learn_junction_score_variants.py`: strict score-only learner;
`scripts/qualify_junction_score_ablation.py`: real-SSD causal
benchmark, matched-recall comparison and zero-write validation;
`scripts/run_junction_score_ablation_ci.sh`: self-hosted preparation
for PubMed or BigANN, preserving the original routing state as a
byte-exact reference.

The workflow uses only the authorized self-hosted geoivf runner and
serializes access to the shared SSD. Do not make paper changes until
the job completes successfully and the full raw artifacts validate
the reference reproduction and all three alternatives.
