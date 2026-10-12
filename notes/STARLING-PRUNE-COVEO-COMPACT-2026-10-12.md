# Starling native component pruning: compact Coveo results

Completed success: [run 38282719359](https://github.com/gilga1983/GeoIVF/actions/runs/38282719359),
native Coveo job 114923700905, artifact
[11762649107](https://github.com/gilga1983/GeoIVF/actions/runs/38282719359/artifacts/11762649107).
Three rotated repetitions, same original-native index, 5K disjoint native
training queries, 20K causal warm-up requests, 5K measured. Frozen native
Starling SSD graph and memory navigator unchanged.

## Matched Recall@10: 97%

| Configuration (B, C, k) | Reads/query | Search latency us | Read saving vs native | Latency saving vs native |
|---|---:|---:|---:|---:|
| Native (0,0,0) | 24.351 | 494.50 | 0% | 0% |
| Full NavHints (14464,2048,8) | 15.011 | 382.93 | 38.35% | 22.56% |
| Junction-only (14464,0,0) | 22.277 | 464.18 | 8.52% | 6.13% |
| Destination-only (0,2048,8) | 15.771 | 381.01 | 35.23% | 22.95% |
| Compact junction-only (4096,0,0) | 23.234 | 474.32 | 4.59% | 4.08% |
| Compact destinations (0,512,2) | 19.993 | 417.39 | 17.90% | 15.59% |
| Compact joint (4096,512,2) | 18.815 | 406.65 | 22.73% | 17.77% |

## Matched Recall@10: 98%

| Configuration (B, C, k) | Reads/query | Search latency us | Read saving vs native | Latency saving vs native |
|---|---:|---:|---:|---:|
| Native (0,0,0) | 30.491 | 596.07 | 0% | 0% |
| Full NavHints (14464,2048,8) | 14.742 | 374.49 | 51.65% | 37.17% |
| Junction-only (14464,0,0) | 28.692 | 567.53 | 5.90% | 4.79% |
| Destination-only (0,2048,8) | 15.250 | 368.80 | 49.99% | 38.13% |
| Compact junction-only (4096,0,0) | 28.846 | 564.79 | 5.39% | 5.25% |
| Compact destinations (0,512,2) | 25.393 | 506.35 | 16.72% | 15.05% |
| Compact joint (4096,512,2) | 25.269 | 513.18 | 17.13% | 13.91% |

## Observations

- On Coveo, dropping all static-junction scoring retains most of the full
  routing gains and slightly improves native query-search time at both
  matched-recall targets, while saving hint memory.
- Reducing BOTH destination history (2048 to 512 IDs) and additional starts
  (8 to 2) materially degrades routing and latency gains, so the responsible
  parameter should be separated in the next budget sweep.
- On this workload, compact joint outperforms compact destinations at 97%
  recall but is slightly slower at 98%; neither matches long history/8 starts.
- The same-session baseline differs from earlier frozen experiments: do not
  compare absolute timings or ratios across epochs as an algorithm effect.
- These are 3 repeats on one query sequence; this pilot does not select a
  universally optimal Starling configuration. Original Starling native search
  timer may exclude post-query feedback; provider reads are logical I/Os.

Next: wait for same-run BigANN compact result and choose the smallest
useful set of components/budgets; preserve the manuscript until full validation.
