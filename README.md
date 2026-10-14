# NavHints: Demand-Aware Navigation for Disk-Resident Vector Search

**Reproducible code and experiment harness for the NavHints PVLDB/VLDB 2027 paper.** This is the code repository, historically named GeoIVF. The manuscript and figures live separately in [gilga1983/DemmandAwareANN](https://github.com/gilga1983/DemmandAwareANN).

**For artifact evaluators:** start with [docs/ARTIFACT_REVIEW.md](docs/ARTIFACT_REVIEW.md). It separates offline checks, frozen-result reproduction and a full real-SSD rerun. Fourteen original result archives, including the complete-request latency study, are listed with SHA-256 digests in [repro/manifest.json](repro/manifest.json). The original GitHub Actions ZIPs expire October 25–27, 2026; a permanent release URL is still required. New Figure 8, junction-scoring, fixed-k memory, and native Starling evidence is cataloged in [the supplemental October 14 manifest](repro/supplemental-manifest-20261014.json).

This `main` branch contains the **final** NavHints Core implementation, source-fidelity checks for competing methods, training scripts, four-dataset SSD replays, optional persistent-exit experiments, and an adapter for the **original** Starling library. Exploratory alternatives remain in their Git branches. The original page-oriented GeoIVF implementation is still present; its previous usage guide is [docs/LEGACY_GEOIVF.md](docs/LEGACY_GEOIVF.md). It is **not** the NavHints paper algorithm.

## Final algorithm

On PubMed1M, Core uses **14,464 learned routing-junction IDs** grouped into 512 PQ-scored regions, a **2,048-ID global unique-FIFO history of successful destinations**, and **up to eight** PQ-scored recent destinations offered as additional starts alongside one learned junction. Training is disjoint; online history is updated **only after each query completes**. Existing DiskANN graph edges, beam, frontier and termination are untouched. Our selector keeps the top ten PQ candidates and admits at most eight to the frontier. The published **102,940-byte** auxiliary-memory figure is the explicitly accounted routing payload, **not total RSS**. Coveo has fewer than 14,464 eligible junctions; the actual learned budget is capped accordingly.

An optional Sample2 persistent-exit extension records up to ten winners for selected junctions, with one-half admission probability and fill-only-empty-slot seeding from the global history. **RAM-only Core is the primary algorithm.** Sample2 is measured separately; logical update counts are not physical write amplification or crash-recovery evidence.

## Activate and reproduce

Requires **Linux**, Python 3.10+ (tested with a Python 3.11 environment), Git, Rust/rustup with **1.97.1**, C/C++ build tools, and access to a **real NVMe SSD** for performance measurements. We use `numpy==2.2.6`, `faiss-cpu>=1.9,<2` where needed, DiskANN commit `fcf90534174cf29c78c9f13b4cccf1fcabff85f5`, and Starling commit `17dc3e8a011533a62374445f53963e951b72883a`. Third-party sources are cloned at their original pinned SHAs, not vendored or modified in this repository.

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
source repro/activate.sh

navhints list                    # View every supported experiment
navhints doctor --mode pubmed    # Check tools and frozen dataset files
navhints run pubmed              # DRY RUN: prints exact command, no SSD work
navhints run pubmed --execute    # HEAVY: pinned index, causal replays, real SSD
```

**No heavy run or dataset download is performed without `--execute`.** Each run archives raw queries/results logs, SHA-256 inputs and relevant source diffs. Experimental scripts use the verified dataset layout under `$HOME/.cache/geoivf`, and refuse missing required inputs rather than silently substituting a dataset. Original Coveo data are licensed: **provide your authorized private frozen source data locally** and do not commit or upload the raw records.

### Available reproduction commands

| Command | Evidence produced |
| --- | --- |
| `navhints run pubmed --execute` | DiskANN / junction-only / final RAM-only Core / Sample2 on PubMed1M |
| `navhints run coveo --execute` | Identical causal stages on chronological Coveo demand |
| `navhints run text2image --execute` | Frozen Text2Image-10M SSD index and four-arm replay |
| `navhints run bigann --execute` | Frozen BigANN-10M SSD index and four-arm replay |
| `navhints run catapult --execute` | Same-session source-aligned Catapult routing vs final Core |
| `navhints run memory --execute` | Fixed **equal-accounted-payload** junction/destination budget ablation |
| `navhints run sensitivity --execute` | Warm-up, controlled locality and training-support shift |
| `navhints run persistence --execute` | Sample1/2/4/10 and empty-slot seeding; reads and logical writes |
| `navhints run starling-coveo --execute` | Final Core in original native Starling on Coveo |
| `navhints run starling-bigann --execute` | Same original Starling port on BigANN-10M |

The original workflows are available in [`.github/workflows/`](.github/workflows/) with `workflow_dispatch`. They are **self-hosted only** (`[self-hosted, linux, x64, geoivf]`) and should not be run simultaneously against the same NVMe device. Use `doctor` first and confirm the correct host, authorized data and free storage. The full source/hardware/data contract is in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

### Rebuild paper figures directly from frozen results

If you only need to **reproduce the published numbers and graphs**, you do **not** need to rerun hours of SSD benchmarks. Download the immutable successful GitHub Actions artifacts and regenerate the CSVs:

```bash
source repro/activate.sh
navhints fetch all             # Preview all 14 evidence downloads
navhints fetch all --execute   # Requires gh CLI and GitHub authentication
navhints verify all           # Validate all 14 ZIP contents and policy fields
navhints_paper                 # Dry run: lists exact paper regeneration/build steps
navhints_paper --execute       # Regenerates read/latency CSVs and tables, then checks LaTeX
```

Clone [the paper repository](https://github.com/gilga1983/DemmandAwareANN) beside GeoIVF as `../DemmandAwareANN` before running the final command. To build its PDF, install its pinned PVLDB 2027 template and local pdfLaTeX/BibTeX tools. The paper regeneration scripts **refuse missing or mismatched artifacts**, never impute values and never extrapolate outside the common recall interval. The `verify-latex` target does not overwrite the tracked PDF; the actual source and plots should be reviewed before preparing a submission PDF.

For a machine without access to large datasets, the fourteen artifact ZIPs are enough for the plotting and prose audit, including the four-dataset complete-request latency subsection. With unmodified original Actions ZIPs, `navhints verify all --original-zip` additionally checks recorded SHA-256 hashes. Frozen original data/SSD graph binaries are **not** checked into Git. See [repro/manifest.json](repro/manifest.json) for engine SHAs, run IDs, artifact names, file layout and policy parameters.

## Directory map

- [`scripts/patch_diskann_hint_ivf_packed_direct.py`](scripts/patch_diskann_hint_ivf_packed_direct.py), [`scripts/patch_diskann_experience_core.py`](scripts/patch_diskann_experience_core.py), [`scripts/patch_navhints_regional_history.py`](scripts/patch_navhints_regional_history.py): final DiskANN routing and online/persistent state patches, applied to the pinned Microsoft checkout.
- [`scripts/learn_global_navigation_landmarks.py`](scripts/learn_global_navigation_landmarks.py), [`scripts/learn_navhints_generic.py`](scripts/learn_navhints_generic.py), [`scripts/build_hint_ivf.py`](scripts/build_hint_ivf.py), [`scripts/build_hint_ivf_generic.py`](scripts/build_hint_ivf_generic.py): workload-trained junctions and compact PQ-region layout.
- [`scripts/qualify_final_core_public.py`](scripts/qualify_final_core_public.py), [`scripts/qualify_final_core_coveo.py`](scripts/qualify_final_core_coveo.py) and their `run_final_core_*_ci.sh` scripts: reproducible, four-arm, causal frozen evaluations.
- [`scripts/qualify_optimal_navhints_catapult.py`](scripts/qualify_optimal_navhints_catapult.py): direct final-Core vs author-aligned Catapult comparison; audited original Catapult source semantics and common DiskANN graph.
- [`scripts/patch_starling_final_core.py`](scripts/patch_starling_final_core.py), [`scripts/generate_starling_final_core.py`](scripts/generate_starling_final_core.py): original Starling source adapter, where its *native* memory navigator and SSD layout remain enabled.
- [`repro/run.py`](repro/run.py), [`repro/activate.sh`](repro/activate.sh), [`repro/manifest.json`](repro/manifest.json): supported activation, preflight, replay, artifact retrieval and paper regeneration.
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): code decisions, exact dataset splits, verified runs and caveats.

## Reproduced headline evidence

All source references are listed in [the frozen manifest](repro/manifest.json). Real-SSD final-Core+Sample2 matched-recall logical-read savings versus unmodified DiskANN are **80.69% Coveo**, **44.23% PubMed**, **34.53% Text2Image-10M**, and **21.01% BigANN-10M** on each run's jointly reachable recall targets; targets differ by workload. On PubMed, RAM-only Core saves **13.28% logical SSD reads** compared with an **author-aligned Catapult hash/LRU routing mechanism on the same pinned DiskANN SSD engine** across 11 common recall anchors, not the author's complete original in-memory system. The Starling port is tested in the **original** author search engine: substantial benefits on Coveo, near break-even logical I/O on BigANN. The repeated query sequences are fixed, so do not infer confidence intervals from independent samples.

All auxiliary-memory budgets exclude some container metadata; SSD read counts and optional logical writes must not be relabeled physical NVMe block operations. Reproducing the paper means reproducing the **evaluation protocol and its limitations**, not merely matching its reported percentages.

## Archival history

This repository began as a page-oriented GeoIVF prototype. Its code (`geoivf/`, `native/`, `tests/`) and earlier workflows are preserved for historical reproducibility but are **not** the selected NavHints paper implementation. The original README is [archived without editing](docs/LEGACY_GEOIVF.md). Explore earlier NavHints alternatives on their named Git branches; `main` supports the selected final Core and the validated comparisons, not every prototype.
