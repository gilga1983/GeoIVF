# NavHints: two-minute artifact review

**Paper:** *NavHints: Demand-Aware Navigation for Disk-Resident Vector Search*
(regular research paper, PVLDB Volume 20 / VLDB 2027).

**Public code:** https://github.com/gilga1983/GeoIVF

## Fast source-level validation (no data or compiled dependencies)

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
python3 -m unittest discover -s tests -p 'test_navhints_repro.py' -v
python3 -m unittest discover -s tests -p 'test_navhints_artifact.py' -v
python3 repro/run.py review
```

This validates the exact selected NavHints controller, causal/no-download
dry-run controls, original-source markers, and 30 immutable experiment
provenance records (14 main + nine supplemental + seven additional). It uses
only Python >=3.10 and standard-library packages. It does not access the
network beyond the initial Git clone and does **not** run SSD workloads.

## Paper result verification (requires reviewer ZIP)

Obtain the separately approved *data-minimized reviewer evidence archive*
from the artifact release. The original raw ZIPs are deliberately omitted
from this reviewer candidate pending rights clearance. From the code checkout:

```bash
python3 repro/run.py review --bundle /path/to/NavHints-PVLDB-evidence.zip
```

The verifier first SHA-256-checks every included aggregate JSON member against
its manifest, then computes the actual values underlying the paper's core
read-saving bars, complete-request latency table, both Figure 8 panels and
junction-ranking sensitivity table. It checks target counts and recall
interpolation without extrapolation, and reports each dataset separately.
On the frozen reviewer candidate this completes in seconds without extra
Python dependencies, proprietary datasets, Linux or NVMe hardware.

**This does not reproduce the raw measurement process.** A source-independent
real SSD rerun requires Linux, the original input datasets and query order,
a pinned DiskANN/Starling checkout, Rust and a suitably isolated SSD.

## Native reproduction (optional, heavy)

```bash
source repro/activate.sh
navhints list
navhints doctor --mode pubmed
navhints run pubmed                  # inspect command, dry run
navhints run pubmed --execute        # real pinned SSD experiment
```

The original runner scripts are included. Never run overlapping latency
benchmarks on the same device, and do not substitute unauthorized Coveo data.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact split paths,
input formats, pinned dependencies, SSD isolation rules and method semantics.

## Which experiment supports which claim?

See the [paper/evidence map](ARTIFACT_MAP.md) and the immutable manifests in
`repro/`. The selected algorithm is RAM-only (B up to 14,464 junction IDs,
512 regions, C=2,048 causal unique-FIFO destinations, k=8 starts). The
optional Sample2 persistent-exit extension is a distinct experiment.

## Current publication status

The public code is available now. The separate data-minimized candidate was
built and checked privately, but requires author approval for redistribution
and permanent archival hosting. The original frozen archives contain machine
paths, diagnostics, and material potentially governed by Coveo usage terms.
The code repository has **no declared top-level software license yet**.

The original paper source repository is private and is not required to run
*either* of the offline validation commands above. Regenerating the complete
LaTeX figure collection from original SSD archives requires an authorized
copy of the paper derivation scripts. See [release readiness](RELEASE-READINESS-2026-10-17.md).
