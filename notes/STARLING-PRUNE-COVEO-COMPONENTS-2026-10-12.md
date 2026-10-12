# Native Starling pruning pilot: Coveo component results (12 Oct 2026)

**Status:** succeeded, preliminary component ablation (NOT a selected configuration
for the paper). Original native Starling, same frozen Coveo SSD index and causal
training/query stream, 3 rotated repetitions.

- [GitHub Actions run 38273803590](https://github.com/gilga1983/GeoIVF/actions/runs/38273803590)
- Completed native Coveo job: 114894356068
- [Raw artifact 11758274527](https://github.com/gilga1983/GeoIVF/actions/runs/38273803590/artifacts/11758274527)
- Pinned original author Starling: 17dc3e8a011533a62374445f53963e951b72883a
- The source checker and native Coveo job both passed.

## Matched Recall@10 at 97%

| Mode | Reads/query | Search latency us | Read savings vs native | Latency savings vs native |
|---|---:|---:|---:|---:|
| baseline | 29.216 | 560.58 | 0.00% | 0.00% |
| final_core (B14464 C2048 k8) | 14.904 | 379.05 | 48.99% | 32.38% |
| junction_only (B14464 C0 k0) | 26.138 | 523.47 | 10.54% | 6.62% |
| destination_only (B0 C2048 k8) | 15.726 | 379.17 | 46.17% | 32.36% |

## Matched Recall@10 at 98%

| Mode | Reads/query | Search latency us | Read savings vs native | Latency savings vs native |
|---|---:|---:|---:|---:|
| baseline | 37.963 | 711.52 | 0.00% | 0.00% |
| final_core | 14.706 | 372.64 | 61.26% | 47.63% |
| junction_only | 34.257 | 659.96 | 9.76% | 7.25% |
| destination_only | 16.525 | 391.18 | 56.47% | 45.02% |

## Interpretation

On Coveo, historical destination starts account for most of the navigation
benefit. Removing the junction directory produces almost identical matched-
recall search latency at 97% recall; at 98% recall the original full configuration
remains modestly faster. The destination-only mode avoids the junction selector
PQ scan entirely, and so is substantially simpler and uses less hint memory.

These results support the user-requested next step of reducing remaining
destination-history size and start count. They do not establish that the
optimized destination-only mode is a universal champion: BigANN/other native
datasets, compact-size variants, variance, matched recall and feedback-inclusive
time still require measurement.

## Protocol caveat

This same-session result is an *independent new experiment* and its absolute
ratios differ from the earlier frozen original-Starling table. The old paper
numbers and main have not been updated. For any new paper comparison, use a
coherent within-session recall grid and verify native query/provenance files.
