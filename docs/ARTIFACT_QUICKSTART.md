# NavHints: reviewer quickstart

**Paper:** NavHints: Demand-Aware Navigation for Disk-Resident Vector Search
(regular research paper, PVLDB Volume 20 / VLDB 2027).

**Public entry point:** https://github.com/gilga1983/GeoIVF

## Verify the recorded results

Python 3.10+ and Git suffice. No package installation, private manuscript
checkout, account login, dataset download or NVMe hardware is needed.

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
make test
python3 repro/run.py review --bundle repro/navhints-reviewer-evidence.zip
```

Expected: 32 required original-archive provenance records; the public ZIP
contains 35 source archives (three optional controls) and 48 checksummed
aggregate JSON outputs. The verifier recomputes the headline four-workload
read savings, four complete-request latency comparisons, all Figure 8
observations, junction scoring, and every Table 5 source/capacity row.
Table 5 values are recomputed from measured recall/read frontiers at six
supported recall targets, including the negative random-ID control.

This is reproduction of numerical analysis from recorded measurements.
It does not independently rerun the original SSD measurements.

## Regenerate plotted data and render figures

```bash
python3 repro/export_paper_results.py --output artifacts/paper
cd artifacts/paper
pdflatex -interaction=nonstopmode -halt-on-error figures.tex
pdflatex -interaction=nonstopmode -halt-on-error figures.tex
```

The Python step regenerates 20 plotted CSVs and verifies their agreement
with the frozen manuscript reference values. It exports the destination
controls, junction scoring, motivation curves and latency details as well.
The TeX sheet uses the eight original figure sources and nine table sources;
the table sources are pinned manuscript outputs, with numerical verification
coverage distinguished in [ARTIFACT_MAP.md](ARTIFACT_MAP.md).

Only the optional PDF step needs TeX Live with TikZ/PGFPlots, booktabs,
multirow, xspace, geometry and colortbl. The private paper repository is
not required. Choose a new output path or use `--overwrite` for another export.

## Re-run physical experiments

```bash
source repro/activate.sh
navhints doctor --mode pubmed
navhints run pubmed                  # command preview
navhints run pubmed --execute        # real SSD experiment
```

Full replay requires Linux, the pinned DiskANN/Starling sources, original
input data and query order, and an otherwise quiet NVMe device. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Original licensed Coveo data are
supplied separately by an authorized user. Do not substitute another dataset
and describe it as reproducing the same workload.

The evidence ZIP omits raw vectors, query text, graph binaries, per-request
profiler logs, host diagnostics and third-party source trees. Original raw
archives have been preserved separately; review does not rely on expiring
GitHub Actions downloads. Permissions for original contributions are in
[LICENSE](../LICENSE); third-party terms are unchanged.
