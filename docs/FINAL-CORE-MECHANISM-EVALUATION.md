# NavHints routing-mechanism evidence (final Core only)

Goal: explain *where* logical SSD reads disappear, separate globally learned junction guidance from successful-destination starts, and identify when eight starts become redundant. Do not reintroduce 16K/512/one-start or report data from it as final-Core evidence.

## Active runs
- Section 4.7 alternative memory allocations: https://github.com/gilga1983/GeoIVF/actions/runs/38164299742 (PubMed and BigANN final-Core reference; verify success before use).
- Number of starts × controlled locality: https://github.com/gilga1983/GeoIVF/actions/runs/38164539106 (PubMed final Core, 0/1/2/4/8 starts, 5 locality populations). One selected 14,464-junction directory, 512 regions, 2,048 recent-ID FIFO. No persistence.

## Next instrumentation step (not yet implemented)
Use the *same frozen queries and graph* to gather per-request and per-expansion traces for the unmodified DiskANN, junction-only (0 additional destinations), and final Core (eight destinations). The following metrics must use engine-native counters rather than infer from observed latency:

1. Actual graph-page reads and unique graph-record IDs expanded, broken into early / middle / last search progress windows. Define windows using percentage of total expansions to avoid assuming equal wall-clock duration. This is a descriptive partition, not a counterfactual proof.
2. PQ distance computations, candidate insertions, duplicate/visited suppressions, and the chosen region plus candidate starts for each query.
3. For each extra start: whether it was inserted; whether it was ever expanded; whether it belonged to the exact ground-truth top 10; and whether it led to a newly expanded result-near vertex. Do **not** attribute causal saved I/O to an individual start based only on expansion ordering.
4. All per-query routing CPU timings, including medoid/group selection, scanning the 2,048 history, frontier handling, and post-completion FIFO feedback. Report logical I/O and complete feedback-inclusive time separately.
5. Optional per-query peak/steady resident bytes attributable to Core IDs, medoids/PQ copies, membership structures and scratch. Accounted payload != RSS.

To establish causal contributions, rerun matched-recall controlled ablations with fixed B=14,464 and C=2,048: zero extra starts (junction-only), 1, 2, 4, 8 starts. For each experiment report the common recall targets jointly reachable by all arms, no extrapolation. Compute relative read savings against unmodified DiskANN and against junction-only, and include adverse cases.

## Paper storyboard
One figure: two or three panels showing matched-recall expansion cost by progress stage (with careful caveat about normalized stages), per-start usefulness/duplicate rates, and changing best k with locality. One overhead table with CPU and memory accounting. Add only after complete run artifacts validate the claims. Results for the retired allocation remain out of the main paper.
