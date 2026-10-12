# Native Starling pruning pilot: BigANN-10M components (12 Oct 2026)

**Status:** completed successfully. This is the first-stage original-native
ablation; the frozen manuscript and its original Starling table are untouched.

- [GitHub Actions run 38273803590](https://github.com/gilga1983/GeoIVF/actions/runs/38273803590)
- [Native BigANN artifact 11761014331](https://github.com/gilga1983/GeoIVF/actions/runs/38273803590/artifacts/11761014331)
- Original pinned Starling source SHA 17dc3e8a011533a62374445f53963e951b72883a
- Frozen native graph/navigator and disjoint training 5K / warm-up 4K /
  measured 1K; three rotated repetitions.

## Matched Recall@10 of 97%

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| Unmodified native Starling | 40.981 | 1070.16 | 0.00% | 0.00% |
| Full NavHints (B14464/C2048/k8) | 40.792 | 1240.60 | +0.46% | -15.93% |
| Junction-only (B14464/C0/k0) | 40.750 | 1109.35 | +0.56% | -3.66% |
| Destination-only (B0/C2048/k8) | 40.972 | 1192.95 | +0.02% | -11.47% |

## Matched Recall@10 of 98%

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| Unmodified native Starling | 46.978 | 1236.74 | 0.00% | 0.00% |
| Full NavHints | 46.442 | 1405.73 | +1.14% | -13.66% |
| Junction-only | 46.029 | 1259.17 | +2.02% | -1.81% |
| Destination-only | 47.092 | 1371.61 | -0.24% | -10.91% |

## Interpretation

- With Starling's own BigANN navigator active, destination-history scoring
  provides negligible read benefit and substantial CPU/search-time overhead.
- Junction-only retains the small but measurable logical-read savings and
  removes much of the full method's latency penalty: about 131 us/query at
  97% recall and 147 us/query at 98% recall.
- This contrasts with Coveo, where destination-only retained almost all
  full-NavHints benefit, reinforcing dataset/system-specific component value.
- Junction-only is **not** yet a latency champion: at both tested recall
  levels it is slightly slower than native Starling. A smaller junction
  directory (B4096) will test whether the remaining CPU penalty can be
  removed while preserving I/O benefits.
- Per-query search latency is measured in Starling's native search timer,
  which may exclude post-query destination feedback. Native SSD reads are
  provider logical operations. Three repetitions reuse a fixed query
  sequence, so these are not independent workload samples.
- Result selection and resizing on the same heldout stream would need
  a separate validation protocol before a final paper champion is claimed.

## Next run

[Compact-budget workflow 38282719359](https://github.com/gilga1983/GeoIVF/actions/runs/38282719359)
starts the predeclared B4096/C512/k2 variants on Coveo and BigANN, alongside
native baseline, frozen full NavHints and component-only controls. The
experiment branch is isolated from main and the LaTeX paper.
