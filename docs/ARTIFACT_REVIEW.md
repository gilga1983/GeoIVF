# NavHints artifact evaluation

This is the **artifact-review entry point** for *NavHints: Demand-Aware Navigation
for Disk-Resident Vector Search*. The paper is maintained separately in
[DemmandAwareANN](https://github.com/gilga1983/DemmandAwareANN). This repository
is historically named GeoIVF; the older `geoivf/`, `native/` and many research
workflows are **not** the algorithm used in the NavHints paper. Keep them for
research provenance, but use the paths below to review NavHints.

**Status (2026-10-13):** Fourteen original GitHub Actions ZIPs are inventoried,
including all four complete-request latency studies. The original Actions
archives expire **October 25–27, 2026**. They must be preserved in a
long-lived, access-controlled archive and assigned a stable review URL before
they expire. This repository currently has no top-level LICENSE; redistribution
terms require the owner's approval. These are release blockers, not resolved
by a successful smoke test.

## 1. What can be evaluated?

| Level | Data, machine and time | Evidence |
| --- | --- | --- |
| A. Static/offline preflight | Linux/macOS/Windows, Python 3.10+, standard library, minutes | Repository completeness, paper controller, provenance hashes, safe CLI behavior |
| B. Frozen-result reproduction | 14 original ZIPs, Python 3.10+, a sibling paper checkout; LaTeX only to rebuild PDF | Validate archived outputs and recompute the reported figures and complete-request latency table without running SSD benchmarks |
| C. Full experiment | Linux, pinned Rust/C++, DiskANN/Starling, frozen datasets, spare RAM/storage, exclusive real NVMe measurement window; substantial wall time | Rebuild learned state, run causal held-out replays and measure the underlying engine |

Do **not** interpret A or B as an independent NVMe rerun. Equally, a new C run
on a different SSD or workload is not a bitwise reproduction of the original
timing numbers. The original successful experiments have input hashes and
patched-source records in their ZIPs.

## 2. Five-minute, no-data start

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
python3 -m unittest discover -s tests -p 'test_navhints_repro.py' -v
source repro/activate.sh
navhints list
navhints run pubmed                    # dry run: does NOT start an SSD benchmark
navhints fetch all                     # dry run: does NOT fetch ZIPs or create files
navhints_paper                         # dry run: lists all derivation/build commands
```

This path uses only the Python standard library and does not install Faiss,
clone third-party engines, fetch restricted data or execute performance tests.
The older `make test` target builds and tests the separate legacy GeoIVF
implementation; it is **not** a smoke test of DiskANN+NavHints.

## 3. Verify original evidence and reproduce the figures

Obtain the frozen review bundle containing the **fourteen original GitHub
Actions ZIPs** from the maintainer's permanent artifact archive. Put the files
in `artifacts/frozen/`, named
`coveo.zip`, `pubmed.zip`, `text2image.zip`, `bigann.zip`,
`catapult.zip`, `memory.zip`, `sensitivity.zip`, `persistence.zip`,
`starling-coveo.zip`, `starling-bigann.zip`, and
`service-{pubmed,coveo,text2image,bigann}.zip`.
The complete original Actions run IDs, artifact IDs, source commits, member
paths and **SHA-256 of the unmodified GitHub ZIP bytes** are in
[`repro/manifest.json`](../repro/manifest.json).

```bash
navhints verify all --original-zip
```

This validates each original ZIP's SHA-256 **and** its internal JSON policy and
producing-source marker. Without `--original-zip`, `navhints verify all`
validates members, controller, method arms, source SHA and causal protocol,
but does not authenticate original compressed ZIP bytes.

If the original archives have **not** yet expired, the existing GitHub CLI
fetcher provides an alternative to the permanent bundle:

```bash
gh auth login
navhints fetch all --execute
navhints verify all
```

`gh run download` extracts files; `navhints fetch` recompresses them into
a canonical per-case ZIP for downstream plotting. **Do not apply
`--original-zip` to these recompressed files.** Original SHA-256 values in
the manifest refer to the Actions API ZIP byte stream, not to this repack.

Clone [DemmandAwareANN](https://github.com/gilga1983/DemmandAwareANN) alongside
this checkout and install its LaTeX dependencies. To regenerate the
four-workload read results, sensitivity plots and **all four complete-request
latency curves/table** from the archived JSON:

```bash
navhints_paper                       # inspect exact commands first
navhints_paper --execute             # updates paper CSV/table files; scratch PDF build
```

For a reviewer wanting only the latency result, without LaTeX or other
figures, run the paper script in a scratch copy of the paper repository:

```bash
python3 ../DemmandAwareANN/scripts/derive_service_latency.py \
  --coveo artifacts/frozen/service-coveo.zip \
  --pubmed artifacts/frozen/service-pubmed.zip \
  --text2image artifacts/frozen/service-text2image.zip \
  --bigann artifacts/frozen/service-bigann.zip \
  --out-data /tmp/navhints-review/data \
  --out-table /tmp/navhints-review/service-table.tex \
  --out-summary /tmp/navhints-review/service-summary.json
```

The derivative validates the per-repetition Pareto frontiers and refuses
unsupported recall interpolation. The generated values can be compared against
`figures/data/eval_service_*.csv` and
`tables/eval_service_latency_table.tex` in the paper checkout. This command
runs on a normal CPU; it does not rerun the SSD experiment.

## 3a. Supplemental evidence added to the current manuscript

The original 14-archive inventory predates later approved experiments. The
additional evidence is now inventoried, with exact run IDs, artifact IDs and
observed original-ZIP SHA-256 digests, in
[the October 14 supplemental manifest](../repro/supplemental-manifest-20261014.json).

It includes both four-curve Figure 8 studies, all four junction-ranking
comparisons, the fixed-k memory controls, and the original native Starling
PubMed and Text2Image experiments. In particular, the original native
Starling runs must **not** be confused with later pruning experiments.

The existing `navhints verify all` command intentionally covers only the
original 14-package inventory. This new manifest is additional evidence;
it is **not** automatically fetched or verified by that original command.
Do not tell reviewers the old 14-package command alone reproduces the
current paper's new sensitivity table. The supplemental ZIPs must be included
and checked in the eventual permanent review bundle.

## 4. Re-run the actual SSD measurements

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for data acquisition, pinned
upstream revisions, exact query splits and engine builds, and
[IMPLEMENTATION.md](IMPLEMENTATION.md) for the RAM-only design and adapter
locations. Key reference code is:

- DiskANN source patches:
  `scripts/patch_diskann_hint_ivf_packed_direct.py`,
  `scripts/patch_diskann_experience_core.py`,
  `scripts/patch_navhints_regional_history.py`
- Trained routing data:
  `scripts/learn_global_navigation_landmarks.py`,
  `scripts/learn_navhints_generic.py`, `scripts/build_hint_ivf*.py`
- Final real-SSD replay: `scripts/run_final_core_*_ci.sh`
- Native Starling port: `scripts/patch_starling_final_core.py`

The original engine sources are fetched at recorded SHAs, then the patches
are applied. No third-party source tree or large SSD graph is vendored in
this repository. A dry run is always available:

```bash
navhints doctor --mode pubmed
navhints run pubmed
```

Only a deliberately authorized real replay should include `--execute`.
Run SSD benchmarks **sequentially** on an otherwise quiet physical NVMe.
The code repository is public, but the original Coveo production data are
licensed and must be supplied separately by an authorized owner; do not
upload that raw dataset to a public archive. PubMed, Text2Image and BigANN
also require exact original vector/order/split/index inputs. Replacing the
datasets changes the scientific question.

## 5. Interpretation and honest boundaries

The primary algorithm is RAM-only NavHints Core:
at most 14,464 learned junction IDs in 512 geometric groups, 2,048 successful
destination IDs in a causal unique-FIFO history, and at most eight PQ-ranked
additional starts. It does not change DiskANN's graph topology or reuse
cached answers. The 102,940-byte PubMed figure is **accounted routing
payload**, not an RSS measurement. Optional sampled persistent exits
(Sample2) are a separate experiment, not a mandatory core component.

- A logical SSD read/page counter is **not** a physical NAND operation.
- Real-request latency is measured separately from logical I/O. Workload
  and recall regime matter; a few tested points can regress.
- Starling retains its native navigator and graph layout. Its BigANN gain
  is small or can regress; do not claim universal performance improvement.
- The comparison called Catapult is a source-aligned **routing-policy port
  inside common disk-resident DiskANN**, not the authors' entire original
  search implementation.
- Repetitions replay the same fixed queries. They are not independent
  request populations or a basis for IID confidence intervals.

## 6. Release checklist and retention

Before depositing an artifact-release archive, complete the following:

- [ ] Preserve the fourteen ZIPs **before October 25–27, 2026** and check
      the original SHA-256 values against `repro/manifest.json`.
- [ ] Review all uploaded ZIPs for embedded absolute paths, personal host
      information, credentials, and licensed Coveo content before public
      distribution. A private archival backup can precede a sanitized
      public release.
- [ ] Publish a permanent artifact URL/DOI (e.g. an institutional repository
      or Zenodo) and record the archived package's checksum.
- [ ] Decide and add an explicit repository software license after verifying
      redistribution rights for original author code and embedded patches.
- [ ] Integrate any later paper-approved complete-service experiments or
      table changes into the selected artifact commit, then freeze and tag it.
- [ ] Recheck the paper's artifact-availability URL, final TeX compile,
      figure consistency, dependency environment and access from a clean
      reviewer checkout.

**This document is not a claim that the last six items are already done.**
It is a deterministic path from the current evidence to an auditable release.
