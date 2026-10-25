# NavHints: Demand-Aware Navigation for Disk-Resident Vector Search

**Code, reviewer evidence and experiment harness for the NavHints PVLDB/VLDB 2027 paper.** This is the code repository, historically named GeoIVF. All reviewer quickstart commands and standalone plot exports work from this public checkout. The authoring manuscript is maintained separately.

**For artifact reviewers:** start with [ARTIFACT_QUICKSTART.md](docs/ARTIFACT_QUICKSTART.md) and the [paper-to-evidence map](docs/ARTIFACT_MAP.md). The [bundled evidence ZIP](repro/navhints-reviewer-evidence.zip) contains 48 aggregate JSON outputs from 35 original experiment archives, including both Table 5 studies. Its SHA-256 is frozen in [the release record](repro/release-20261025.json). Original raw archives are preserved separately. See [current release status](docs/RELEASE-READINESS-2026-10-25.md) and [review/reproduction permissions](LICENSE).

This `main` branch contains the **final** NavHints Core implementation, source-fidelity checks for competing methods, training scripts, four-dataset SSD replays, optional persistent-exit experiments, and an adapter for the **original** Starling library. The maintained tree contains NavHints and its paper comparisons. The earlier GeoIVF prototype and retired experiments are recoverable from Git history, including commit `213db3766d27833fb6975fb2fccf5ca02c4d6f1f`.

## Final algorithm

On PubMed1M, Core uses **14,464 learned routing-junction IDs** grouped into 512 PQ-scored regions, a **2,048-ID global unique-FIFO history of successful destinations**, and **up to eight** PQ-scored recent destinations offered as additional starts alongside one learned junction. Training is disjoint; online history is updated **only after each query completes**. Existing DiskANN graph edges, beam, frontier and termination are untouched. Our selector keeps the top ten PQ candidates and admits at most eight to the frontier. The published **102,940-byte** auxiliary-memory figure is the explicitly accounted routing payload, **not total RSS**. Coveo has fewer than 14,464 eligible junctions; the actual learned budget is capped accordingly.

An optional Sample2 persistent-exit extension records up to ten winners for selected junctions, with one-half admission probability and fill-only-empty-slot seeding from the global history. **RAM-only Core is the primary algorithm.** Sample2 is measured separately; logical update counts are not physical write amplification or crash-recovery evidence.

## Two-minute reviewer preflight (Python standard library only)

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
python3 -m unittest discover -s tests -p 'test_navhints_repro.py' -v
python3 -m unittest discover -s tests -p 'test_navhints_artifact.py' -v
python3 repro/run.py review
python3 repro/run.py review --bundle repro/navhints-reviewer-evidence.zip
python3 repro/export_paper_results.py --output artifacts/paper
```

The manifest-only command verifies 32 required experiment provenance records;
the bundle command checks each aggregate JSON's SHA-256 and independently
recomputes the headline read savings, full-request latency, Figure 8 and
junction-scoring results and every Table 5 row at the original matched-recall grids. **Neither
command downloads datasets, invokes a simulator, compiles DiskANN, or touches
the SSD.** The plot exporter regenerates 20 plotted CSVs, checks their numerical agreement with the frozen manuscript, and prepares the eight figures and nine tables without the private manuscript repository. Rendering the sheet requires pdfLaTeX, TikZ/PGFPlots, booktabs and multirow:

```bash
cd artifacts/paper
pdflatex -interaction=nonstopmode -halt-on-error figures.tex
pdflatex -interaction=nonstopmode -halt-on-error figures.tex
```

These are checks of recorded measurements. Full physical experiment, hardware and data requirements are below.

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

### Regenerate figures from the public reviewer package

```bash
python3 repro/export_paper_results.py --output artifacts/paper
```

This is the supported public plotting path. It needs only this checkout and
Python's standard library. See the quickstart for optional PDF rendering.
The original manuscript's derivation scripts and 20 independent reference
CSVs are bundled under `repro/paper/` with their source revision and hashes.
The exact source archives and input datasets required for new SSD experiments
remain a separate layer, documented in `docs/REPRODUCIBILITY.md`.

## Directory map

For a source-level walkthrough, patch-layer order, archived identifier meanings
and contributor checks, see [CODE_GUIDE.md](docs/CODE_GUIDE.md).

- [`scripts/patch_diskann_hint_ivf_packed_direct.py`](scripts/patch_diskann_hint_ivf_packed_direct.py), [`scripts/patch_diskann_experience_core.py`](scripts/patch_diskann_experience_core.py), [`scripts/patch_navhints_regional_history.py`](scripts/patch_navhints_regional_history.py): final DiskANN routing and online/persistent state patches, applied to the pinned Microsoft checkout.
- [`scripts/learn_global_navigation_landmarks.py`](scripts/learn_global_navigation_landmarks.py), [`scripts/learn_navhints_generic.py`](scripts/learn_navhints_generic.py), [`scripts/build_hint_ivf.py`](scripts/build_hint_ivf.py), [`scripts/build_hint_ivf_generic.py`](scripts/build_hint_ivf_generic.py): workload-trained junctions and compact PQ-region layout.
- [`scripts/qualify_final_core_public.py`](scripts/qualify_final_core_public.py), [`scripts/qualify_final_core_coveo.py`](scripts/qualify_final_core_coveo.py) and their `run_final_core_*_ci.sh` scripts: reproducible, four-arm, causal frozen evaluations.
- [`scripts/qualify_optimal_navhints_catapult.py`](scripts/qualify_optimal_navhints_catapult.py): direct final-Core vs author-aligned Catapult comparison; audited original Catapult source semantics and common DiskANN graph.
- [`scripts/patch_starling_final_core.py`](scripts/patch_starling_final_core.py), [`scripts/generate_starling_final_core.py`](scripts/generate_starling_final_core.py): original Starling source adapter, where its *native* memory navigator and SSD layout remain enabled.
- [`repro/run.py`](repro/run.py), [`repro/activate.sh`](repro/activate.sh), [`repro/manifest.json`](repro/manifest.json): supported activation, preflight, replay, artifact retrieval and paper regeneration.
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): code decisions, exact dataset splits, verified runs and caveats.

## Reproduced headline evidence

For the **RAM-only NavHints Core presented in the paper**, mean matched-recall logical SSD-read savings versus unmodified DiskANN are **80.15% Coveo, 44.13% PubMed, 34.57% Text2Image-10M, and 21.03% BigANN-10M**. The distinct target grids contain 4/5/5/4 points. The **optional Sample2 persistent-exit** arm has *different* Full-anchored summary results (80.69%, 44.23%, 34.53%, and 21.01%); these are not the main paper's Core figures and should not be substituted for them. The original records are in [the frozen manifest](repro/manifest.json), and the current summary is in [repro/current-paper-summary-20261017.json](repro/current-paper-summary-20261017.json). On PubMed, RAM-only Core saves **13.28% logical SSD reads** compared with an **author-aligned Catapult hash/LRU routing mechanism on the same pinned DiskANN SSD engine** across 11 common recall anchors, not the author's complete original in-memory system. The Starling port is tested in the **original** author search engine: substantial benefits on Coveo, near break-even logical I/O on BigANN. The repeated query sequences are fixed, so do not infer confidence intervals from independent samples.

All auxiliary-memory budgets exclude some container metadata; SSD read counts and optional logical writes must not be relabeled physical NVMe block operations. Reproducing the paper means reproducing the **evaluation protocol and its limitations**, not merely matching its reported percentages.

## Build and repository scope

```bash
make test         # offline NavHints tests, including native replay generation
make review       # verify the bundled aggregate evidence
make paper-data   # regenerate plotted CSVs into artifacts/paper
make figures      # optional two-pass pdfLaTeX figure/table sheet
```

The maintained tree contains `scripts/` (NavHints, training, engine adapters
and paper comparisons), `repro/` (frozen evidence and plotting snapshot),
`tests/` (offline source/reproduction checks), and `docs/` (current reviewer
instructions). `requirements.txt` is for training and physical replays;
o third-party Python package is needed for `make test` or `make review`.

The native Starling generator composes several strict source transformations.
Its generator/patch helpers remain required even when their filenames refer
to earlier controls. Four generated final replay scripts are checked against
the pre-cleanup source hashes. The removed GeoIVF package, custom native reader,
legacy tests, benchmarks and workflows remain recoverable in Git history.
The repository URL, runner labels and frozen dataset-cache paths are retained
for compatibility with the paper and original experiment inputs.
