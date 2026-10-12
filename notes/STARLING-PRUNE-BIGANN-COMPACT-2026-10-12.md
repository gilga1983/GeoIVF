# Original native Starling: BigANN-10M compact-budget ablation

**Status: completed and verified.** Independent compact-stage native replay from
[GitHub Actions run 38282719359](https://github.com/gilga1983/GeoIVF/actions/runs/38282719359),
self-hosted native job 114923701017, archive
[11771247974](https://github.com/gilga1983/GeoIVF/actions/runs/38282719359/artifacts/11771247974).
The source-check, Coveo, and BigANN jobs all succeeded. Unmodified pinned
Starling graph, SSD page layout, in-memory navigator and 3 rotated
repetitions on a fixed 5K disjoint training / 4K warm / 1K measured BigANN
query stream, native uint8 L2, same native recall anchor interpolations.

## At 97% matched Recall@10

| Mode (B,C,k) | Reads/query | Search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native (0,0,0) | 41.391 | 1130.45 | 0.00% | 0.00% |
| full (14464,2048,8) | 41.033 | 1304.66 | 0.87% | -15.41% |
| junction_only (14464,0,0) | 41.118 | 1170.29 | 0.66% | -3.52% |
| destination_only (0,2048,8) | 41.287 | 1244.91 | 0.25% | -10.13% |
| junction_compact (4096,0,0) | 41.253 | 1156.86 | 0.33% | -2.34% |
| destination_compact (0,512,2) | 41.405 | 1152.94 | -0.03% | -1.99% |
| joint_compact (4096,512,2) | 41.296 | 1187.71 | 0.23% | -5.07% |

## At 98% matched Recall@10

| Mode (B,C,k) | Reads/query | Search latency us | Reads saved vs native | Latency saved vs native |
|---|---:|---:|---:|---:|
| native (0,0,0) | 47.981 | 1320.47 | 0.00% | 0.00% |
| full (14464,2048,8) | 47.362 | 1484.49 | 1.29% | -12.42% |
| junction_only (14464,0,0) | 47.195 | 1352.89 | 1.64% | -2.46% |
| destination_only (0,2048,8) | 48.139 | 1444.94 | -0.33% | -9.43% |
| junction_compact (4096,0,0) | 47.650 | 1345.22 | 0.69% | -1.87% |
| destination_compact (0,512,2) | 48.088 | 1350.10 | -0.22% | -2.24% |
| joint_compact (4096,512,2) | 47.885 | 1385.09 | 0.20% | -4.89% |

## Meaning

The best fully native Starling latency at the two reported recall anchors
still comes from unmodified Starling. Pruning destination scoring removes most
of the full NavHints penalty. Shrinking B further from 14,464 to 4,096 IDs
brings the junction-only latency overhead down to 1.87–2.34%, but also lowers
its small I/O gain. The marginal I/O opportunity offered by NavHints is
small relative to the native navigator's existing effectiveness on BigANN.

Contrasting native Coveo in the same compact stage, destination-only retains
nearly all of full NavHints search-latency benefit on Coveo, while the joint
B4096/C512/k2 variant loses substantial gains. This motivates preserving the
useful component and selecting an appropriate budget independently.

**Next:** component ablations on original-native PubMed and Text2Image-10M,
using their own frozen 5K/4K/1K splits and float32 MIPS. Do not add a
dataset-derived automatic gate based on held-out results. An adaptation or
deployment claim requires independent validation.

No frozen manuscript/paper data, main branch, competitor measurements, graph
layout or author Starling search algorithms were changed by this experiment.
