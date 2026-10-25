# NavHints artifact review

This public repository is the reviewer entry point for *NavHints:
Demand-Aware Navigation for Disk-Resident Vector Search*. Start with the
[two-minute quickstart](ARTIFACT_QUICKSTART.md).

## Public review package (25 October 2026)

- Final RAM-only NavHints implementation and source-aligned competitor checks.
- Causal four-workload DiskANN replay scripts and original Starling adapter.
- [Evidence ZIP](../repro/navhints-reviewer-evidence.zip): 48 aggregate JSON
  outputs from 35 checksum-frozen original experiment archives.
- [Release record](../repro/release-20261025.json): evidence ZIP SHA-256,
  code base revision and exact manuscript revision.
- [Paper/evidence map](ARTIFACT_MAP.md): every current figure and result table.
- [Plotting snapshot](../repro/paper/): original figure/table sources,
  derivation scripts and independent reference CSVs from the frozen manuscript.
- [Review and reproduction permission](../LICENSE) for original contributions.

The 32 required original experiment records are in four manifests: 14 main,
nine supplemental, seven motivation/memory studies, and the two recovered
Table 5 studies. The remaining three source archives are optional controls.
Every published aggregate is individually checksummed in the ZIP manifest.
The original raw archives are separately preserved, including both newly
recovered Table 5 originals. No expiring Actions download is needed for the
public quickstart.

## Evaluation levels

| Level | Requirements | What it verifies |
| --- | --- | --- |
| Source preflight | Python 3.10+, minutes | Final controller, provenance and safe command previews |
| Recorded numerical reproduction | Same public checkout, bundled ZIP | Headline reads, complete-request latency, Figure 8, junction scoring, Table 5 |
| Plot regeneration | Same public checkout; TeX optional | 20 independently regenerated plotted CSVs match manuscript references; standalone figure/table sheet |
| Physical SSD replay | Linux, exact authorized datasets, pinned engines, exclusive NVMe window | New measurements using the documented original protocol |

```bash
python3 repro/run.py review
python3 repro/run.py review --bundle repro/navhints-reviewer-evidence.zip
python3 repro/export_paper_results.py --output artifacts/paper
```

The first command checks provenance only. The second verifies the aggregate
bytes and recomputes the indicated numerical results. The third validates
the frozen ZIP checksum, recomputes plotting data, and copies pinned original
plot/table sources into an independent build directory. The source manuscript
repository is not required for any of these public reviewer paths.

## Full physical reproduction

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for dataset contracts, engine
builds and query splits, and [IMPLEMENTATION.md](IMPLEMENTATION.md) for adapter
locations. Review these command previews before executing a physical replay:

```bash
source repro/activate.sh
navhints list
navhints doctor --mode pubmed
navhints run pubmed
```

The runner scripts pin upstream revisions and apply the original patches.
Use `--execute` only with the exact inputs and an isolated measurement window.
Do not overlap benchmarks on the same SSD. The original data, indexes and
third-party engines are acquired separately. Public benchmark access and
Coveo usage terms remain independent of the source-code permission.

For authors retaining the fourteen unmodified original ZIPs, the existing
`navhints verify all --original-zip` checks their exact compressed-byte hashes.
That legacy command covers the original 14-package inventory only; the public
reviewer verifier covers all four current manifests. `navhints_paper` remains
an authoring integration for a manuscript checkout; public plot regeneration
uses `repro/export_paper_results.py` instead.

## Scientific scope

The selected algorithm uses up to 14,464 junction IDs in 512 regions, a
2,048-ID causal unique-FIFO destination history, and up to eight PQ-ranked
extra starts. Table 5 deliberately fixes 16K junctions and one extra start to
isolate destination quality/capacity; it is not a headline configuration.
Optional Sample2 persistent exits are a separate experiment.

Logical SSD-provider reads are not physical NAND operations. The 102,940-byte
figure accounts for routing payload, not RSS. Catapult is a source-aligned
routing-policy port in the common DiskANN engine. Native Starling retains its
own navigator and timing scope, and its adverse BigANN results are preserved.
Repetition runs replay fixed query sequences and are not independent workload
samples. Numerical checks of archived aggregates do not certify fresh SSD
latency on another machine.

## Maintained source tree

The current tree contains NavHints, its reproducibility tools and the
comparison/ablation code needed by the paper. The original GeoIVF library,
custom native-reader experiments, retired experiment documents and their
workflows/tests have been removed from the maintained tree. They remain in
Git history. `make test` and `make review` now run NavHints-only checks.
