# Line representation qualification, 2026-09-29

## First completed experiment

Source commit: e3c8e51253c5ead01bcec974afa882009d297748.
Successful workflow run: 36530008273; artifact: 11016481981.
Artifact SHA256: 37835e5f43d7305a806b668f555edf8536a45b58321e50306dd6f3a1401eaabb.

Canonical TexMex SIFT1M, 1M original FP32 128D vectors, 4 KiB pages, eight
vectors/page, nlist=1024, nprobe=64, k=10, seed=12345. Same 64 development queries
and preassigned routes as preceding experiments. PCA uses the independent learning
file, not queries. These are requested page/extents from RAM replay, not SSD
traffic, latency, throughput, or a CLIP comparison.

Original payload SHA256:
8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7.
PCA basis fingerprint:
d45ac257039d6f7902499e5233bb93d3b1cfeaf581908c39da27e4d5da7b8a64.
Candidate-route SHA256:
16873c69014d6e83c932414fbd76352e9865608b69b2a0bef3a5d776a1e97e4b.
These match the prior dependent-center study.

Each page's line is freely fitted by local PCA. Quantized origins/directions are
decoded before building scalar intervals. The shell bound retains along-line
position and perpendicular norm, whereas the matching ball bound encloses this
region in an ordinary ball. The complete encoding includes two 64-byte vectors,
32 bytes of scalar codes and 12 bytes of scalar quantization parameters: 172
geometry bytes/page. Add the common 8-byte radial interval and shared/structural
metadata. FP32 axes cost 556 bytes/page; 128D quantized axes cost 300.

The adjacent-page refinement performed four alternating passes, preserving page
capacity and IVF membership. It accepted 11800, 10177, 1219 and 580 page-pair
reassignments. 40582 of 125445 pages (32.35%) changed membership. The acceptance
objective is sum(rho^2)+max(rho^2)+0.1*axial_span^2 after refitting; it is not a
query-performance optimum. Partial pages remain fixed.

## Results

Common baseline: 8394.796875 pages/query; 65.09375 whole-list extents/query.
All filtered methods use 64-page planning windows, zero-gap coalescing and the
same deterministic radial bound. Geometry excludes the common radial interval.

| Layout | Encoding | Geometry B/page | Pages/query | Page reduction | Median reduction | Extents/query |
|---|---|---:|---:|---:|---:|---:|
| Original GeoPack | Independent PCA18, 8-bit | 176 | 5732.0625 | 31.7189% | 18.5084% | 679.9375 |
| Original GeoPack | Independent PCA64, 4-bit | 288 | 2189.4844 | 73.9186% | 76.5419% | 838.2031 |
| Original GeoPack | Independent PCA64, 6-bit | 416 | 466.0625 | 94.4482% | 95.6050% | 249.9688 |
| Original GeoPack | Independent PCA64, 8-bit | 544 | 297.9844 | 96.4504% | 97.0974% | 156.6406 |
| Original GeoPack | Quantized line64, ball bound | 172 | 7817.3438 | 6.8787% | 0.4623% | 313.1094 |
| Original GeoPack | Quantized line64, shell bound | 172 | 7754.8438 | 7.6232% | 0.5245% | 328.0938 |
| Original GeoPack | FP32 line64, shell bound | 556 | 7755.1094 | 7.6200% | 0.5312% | 327.6563 |
| Original GeoPack | Quantized line128, shell bound | 300 | 7769.6406 | 7.4469% | 0.7146% | 329.2813 |
| Adjacent-line refinement | Independent PCA18, 8-bit | 176 | 5706.3125 | 32.0256% | 19.0257% | 697.6563 |
| Adjacent-line refinement | Independent PCA64, 4-bit | 288 | 2169.2656 | 74.1594% | 76.8411% | 848.6406 |
| Adjacent-line refinement | Quantized line64, ball bound | 172 | 7776.4531 | 7.3658% | 0.6017% | 331.0156 |
| Adjacent-line refinement | Quantized line64, shell bound | 172 | 7710.4063 | 8.1526% | 0.7012% | 347.4375 |
| Adjacent-line refinement | Quantized line128, shell bound | 300 | 7724.7188 | 7.9821% | 0.9393% | 349.1719 |

The stronger shell bound helps but does not approach the independently encoded
centers at similar memory. At a 176-byte geometry budget, both the 172-byte line
and the 176-byte PCA18 baseline fit; the latter prunes substantially more pages.
Increasing axis precision or removing the initial projection truncation does
not rescue the line representation on these layouts. Tiny differences between
FP32 and quantized axes are possible because they define slightly different
lines and therefore different scalar maps; there is no dominance guarantee.

The freely fitted PCA64 line explains a mean 30.983% of within-page variation
(median 29.764%) on original pages. After adjacent refinement this reaches 32.184%
(mean) and 30.750% (median). Mean perpendicular distance changes only from 140.151
to 138.035. The pages are therefore still thick residual clouds, not thin lines.
Loss of residual angular information is the likely important limitation here;
this is an interpretation supported by the diagnostics, not a separate causal
proof or an impossibility result for other grouping algorithms.

Complete allocated directory arrays: line64 30.07 MiB, line128 45.39 MiB,
FP32 line64 76.01 MiB. Runtime and decoded query scratch are extra. A no-filter
row in raw results still loads legacy summaries, so it is not a minimal-RAM
implementation; its raw geometry field must not be read as necessary overhead.

## Correctness and validation

95 tests collected on the runner: 89 passed and 6 optional io_uring tests skipped
in this synchronous-build workflow. All 13 new local-line tests passed. The
prior separately validated asynchronous backend is unchanged.

All 1600 configuration-query comparisons across 25 settings matched the
independent FP64 candidate-set oracle. There are 64 distinct queries, not 1600
independent samples. All settings have Recall@10=0.99375. The first two queries
per setting supplied 155792 audited rejection decisions; all were safe. These
include repeated pages across variants and are not distinct pages/queries.

All original points were checked against decoded scalar intervals in each of
five line encodings. Independent-center controls perform their existing
all-point coverage audit. Numeric guards remain engineering safeguards, not
formally verified interval arithmetic.

After downloading, every CSV row count, query ID, aggregate mean, memory-component
sum, byte/page relation, candidate-route hash and shell-versus-ball page count
was independently checked. Original-layout independent baselines reproduce the
preceding study. A local test pass also checked 864 bounds across tiny/large
coordinate scales and thin/thick synthetic geometry.

## Scope of the follow-up

The first experiment only exchanges vectors between adjacent pages. A separate
line-first 64-vector-pool construction has been implemented at commit
0b1a9f4590fe4b356c6e867f4f0fa8e409870cc8; its results are not included above.
No conclusion about arbitrary curves, globally optimal line grouping, shared
line dictionaries, or multiple affine basis directions follows from this study.
