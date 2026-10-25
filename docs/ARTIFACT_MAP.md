# NavHints paper to artifact evidence map

This file maps the **current** PVLDB v20 manuscript to immutable experiment
records. A method used in an ablation does not redefine the selected NavHints
algorithm, and different matched-recall grids must not be pooled.

The main reproducibility entry point is [ARTIFACT_REVIEW.md](ARTIFACT_REVIEW.md).
The public aggregate evidence is bundled at `repro/navhints-reviewer-evidence.zip`.
Verify it with `python3 repro/run.py review --bundle repro/navhints-reviewer-evidence.zip`.
The ZIP includes 48 aggregate JSONs from 35 original source archives.
`python3 repro/export_paper_results.py --output artifacts/paper` regenerates
all 20 plotted CSVs and prepares standalone figure/table rendering.

| Manuscript item | Frozen evidence | Reviewer verification |
| --- | --- | --- |
| Figure 1, traversal hotspots | `motivation-junction-concentration` from [additional manifest](../repro/active-figures-table7-sources-20261017.json) | Aggregate training/held-out visit share by workload |
| Figure 2, recent-destination overlap | `motivation-destination-reuse` in additional manifest | Historical oracle overlap, not cached-answer hit rate |
| Figures 3–4 and Algorithm 1 | Source-level design in the paper; final routing implementations `scripts/patch_navhints_regional_history.py`, `scripts/patch_diskann_experience_core.py` | Review invariants and native start-point integration; no measured-data component |
| Figure 5, SSD-read savings | `coveo`, `pubmed`, `text2image`, `bigann` from [original manifest](../repro/manifest.json) | Recompute *Core* matched-recall mean read savings over 4/5/5/4 targets |
| Figure 6, recall–SSD-read frontiers | Same four original Core artifacts | Plot measured per-L recall/read operating points, not extrapolated curves |
| Table 3, Catapult routing | `catapult` from original manifest | Eleven common PubMed targets, source-aligned routing port on DiskANN, not native CatapultDB whole system |
| Figure 7 and Table 4, complete-request latency | `service-pubmed`, `service-coveo`, `service-text2image`, `service-bigann` from original manifest | Non-dominated per-repetition/seed curves; interpolation without extrapolation |
| Table 5, destination-source and size controls | [Two original PubMed studies](../repro/destination-controls-sources-20261025.json), runs 37728823486 and 37672729944; fixed 16K junctions and one extra start | Recompute all three source controls and six capacities from recall/read frontiers at six supported targets; original ZIP digests and exact source SHAs frozen |
| Table 6, fixed-budget memory | `fixed-k-memory-and-review-gates` from [supplemental manifest](../repro/supplemental-manifest-20261014.json) | B+C=16,512 (102,940 charged payload bytes), k=8 controls and a separate k=4 control |
| Table 7, junction-ranking sensitivity | All four `junction-score-*` entries in supplemental manifest | Compare sum-position score with visits and mean position; Coveo uses a separate binding B=8192 |
| Figure 8(a), online warm-up | `figure8-warmup-starts` from supplemental manifest | 7 history lengths × k=1/2/4/8; six recall targets each |
| Figure 8(b), repetition | `figure8-locality-starts` from supplemental manifest | 5 controlled working sets × k=1/2/4/8; changes query identity mix as well as repetition |
| Table 8, alternative uses of RAM | `rq3-*` entries in [additional manifest](../repro/active-figures-table7-sources-20261017.json), plus frozen Core reference | Native BFS cache, workload-hot cache and QSEV at approximately 1x/4x/16x *accounted* memory, not whole RSS |
| Table 9, native Starling | `starling-coveo` and `starling-bigann` in original manifest, `native-starling-pubmed` and `native-starling-text2image` in supplemental manifest | Native Starling search timings may exclude post-query updates; BigANN regression remains visible |

## Evidence tiers

**Level A, source contract:** `python3 repro/run.py review` validates the
four public source manifests, checks unique original artifact digests, and
requires the four workload results and four scoring comparisons in the
paper's aggregate manifest. No nonstandard Python packages are required.

**Level B, archived numerical reproduction:** `python3 repro/run.py review
--bundle <reviewer-bundle.zip>` verifies every selected aggregate member's
SHA-256 and reproduces four principal read-saving results, four complete-
request latency comparisons, 48 Figure 8 observations, and four
junction-scoring ablations, plus all Table 5 controls. This checks the **sanitized result data**, not
bitwise replay of original SSD measurements. Original full ZIPs remain
separately checksum-frozen and privately preserved.

**Level C, full physical replay:** `source repro/activate.sh`, inspect
`navhints doctor --mode ...` and `navhints run ...` first, then deliberately
run `navhints run <experiment> --execute` on Linux with the exact licensed
inputs, pinned DiskANN/Starling revisions and exclusive NVMe device. This
is a substantially heavier procedure. Unavailable licensed data must not
be replaced with unrelated input and called the same experiment.

## Important boundaries

- The core algorithm is RAM-only: up to 14,464 junctions in 512 regions,
  global unique FIFO of 2,048 destinations, k=8. Sample2 persistent exits
  are a separate experiment and its results are not the headline RAM-only
  Core values.
- Logical SSD-provider page reads are not physical NAND operations.
- The 102,940-byte figure is *accounted incremental routing payload*, not
  an assertion of equal process RSS or an overall memory limit.
- Rotated repetitions share frozen queries, so they are technical replays,
  not independent request-population samples.
- Input dataset binaries, the upstream engines, private host diagnostics,
  licensed Coveo source data and raw paper experiment logs are not embedded
  in the lightweight public repository.
- The original authoring manuscript repository remains private. Public numerical
  checks and plot regeneration use the bundled plotting snapshot and do not
  need access to that repository. The standalone TeX sheet preserves the
  original table outputs; independent numeric checks cover the headline,
  latency, destination-control and junction-score results. Other tables remain
  inspectable against the included aggregate evidence and derivation sources.

## Release state

Table 5 provenance has been recovered and verified against every printed
value. The approved public aggregate ZIP, source/plot snapshots, review
permissions and exact checksums are now in the public repository. Raw input
data and raw machine logs are excluded. See
[release status](RELEASE-READINESS-2026-10-25.md).
