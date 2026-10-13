# Original Starling NavHints pruning: PubMed and Text2Image components

**Verified October 13, 2026**. Original native Starling on float32 MIPS, 5K
disjoint training prefix, 4K causal online warm-up, 1K measured, three rotated
repetitions, identical within-dataset SSD graph, navigator and query stream.

- [Successful original-native MIPS workflow #38303625887](https://github.com/gilga1983/GeoIVF/actions/runs/38303625887)
- [PubMed native archive #11776253003](https://github.com/gilga1983/GeoIVF/actions/runs/38303625887/artifacts/11776253003)
- [Text2Image native archive #11781867912](https://github.com/gilga1983/GeoIVF/actions/runs/38303625887/artifacts/11781867912)
- Source check, PubMed and Text2Image Jobs all passed.
- All comparisons interpolate within jointly supported native Recall@10 targets.

## PubMed: matched 99.0% Recall@10

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native | 52.038 | 1524.27 | 0.00% | 0.00% |
| full B14464/C2048/k8 | 27.841 | 1499.75 | 46.50% | 1.61% |
| junction-only B14464 | 47.457 | 1489.33 | 8.80% | 2.29% |
| destination-only C2048/k8 | 32.781 | 1557.09 | 37.01% | -2.15% |

## PubMed: matched 99.5% Recall@10

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native | 82.150 | 2313.46 | 0.00% | 0.00% |
| full B14464/C2048/k8 | 41.063 | 1846.89 | 50.01% | 20.17% |
| junction-only B14464 | 78.467 | 2301.10 | 4.48% | 0.53% |
| destination-only C2048/k8 | 42.984 | 1824.36 | 47.68% | 21.14% |

## Text2Image-10M: matched 60.0% Recall@10

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native | 86.994 | 2172.84 | 0.00% | 0.00% |
| full B14464/C2048/k8 | 83.090 | 2171.85 | 4.49% | 0.05% |
| junction-only B14464 | 84.306 | 2135.29 | 3.09% | 1.73% |
| destination-only C2048/k8 | 83.815 | 2131.70 | 3.65% | 1.89% |

## Text2Image-10M: matched 80.0% Recall@10

| Mode | Reads/query | Native search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native | 282.835 | 7024.63 | 0.00% | 0.00% |
| full B14464/C2048/k8 | 276.145 | 6900.67 | 2.37% | 1.76% |
| junction-only B14464 | 277.322 | 6925.96 | 1.95% | 1.40% |
| destination-only C2048/k8 | 278.758 | 6972.33 | 1.44% | 0.74% |

## Interpretation

- PubMed: at 99.5% Recall@10, destinations account for most benefit, and
  destination-only runs slightly faster than the full configuration. At 99%
  recall, junction-only has slightly better native search time than full
  NavHints; destination-only increases search time.
- Text2Image: gains are modest. Both individual components can outperform
  full NavHints in native search latency at the 60% recall point; at 80%
  recall, the full configuration is marginally best.
- BigANN results from a separate experiment favor junction-only over
  destination-only, with unmodified Starling still faster. Coveo strongly
  favors destination-only and the full C2048/k8 budget.

**Caution:** these repeated native runs share one frozen query sequence,
not independent query samples. Search latency across separate experiment
sessions differs, so do not infer a method effect from old-vs-new timing.
The published/frozen original-Starling paper numbers are unchanged.
This is an exploratory ablation; architecture-specific tuning should use
disjoint validation queries before a final chosen method is claimed.

## Follow-up

Evaluate independently reduced component budgets within native Starling
and at each workload's supported matched recall targets. Explicitly
separate destination history capacity (C) from extra start count (k)
before drawing a conclusion about which memory/CPU budget matters more.
