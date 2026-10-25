# Reproduce the final NavHints paper

This is the **supported execution specification** for the final NavHints Core, not an overview of discarded prototypes. It connects source snapshots, training data, runs, archived artifacts, tables and figures. Commands are safe by default until you add `--execute`.

## 0. Offline numerical validation first

The original paper's headline RAM-only Core experiment can be checked without
access to the private manuscript repository or a real SSD:

```bash
python3 -m unittest discover -s tests -p 'test_navhints_artifact.py' -v
python3 repro/run.py review
python3 repro/run.py review --bundle repro/navhints-reviewer-evidence.zip
python3 repro/export_paper_results.py --output artifacts/paper
```

The data-minimized evidence ZIP and standalone plotting sources are included
in this public checkout. See [ARTIFACT_QUICKSTART.md](ARTIFACT_QUICKSTART.md) and
[ARTIFACT_MAP.md](ARTIFACT_MAP.md). The public aggregate package does not depend on expiring Actions ZIPs;
original raw archives have been preserved separately.

## 1. Check out the code repository

```bash
git clone https://github.com/gilga1983/GeoIVF.git
cd GeoIVF
source repro/activate.sh
navhints list
navhints doctor --mode pubmed
```

Use Linux with an SSD suitable for measured disk-resident graph search, four isolated cores, enough free RAM and disk space for the target dataset and Rust/C++ compilation. The successful source campaigns used self-hosted `geoivf` runners, and **all repeated SSD phases must be serialized on a shared device**. Ten self-hosted runner processes on one laptop do not provide ten independent NVMe devices.

Required tools: Bash, Git, Python 3.10+, `flock`, `sha256sum`, `rustup`, Rust **1.97.1**, Cargo, and Linux build tools. Python code uses NumPy **2.2.6** and Faiss CPU `>=1.9,<2` where applicable. External dependencies are installed in isolated environments; the code repository does not vendor their original source. The Microsoft DiskANN checkout is pinned to `fcf90534174cf29c78c9f13b4cccf1fcabff85f5` with a locked Cargo build; native Starling is pinned to `17dc3e8a011533a62374445f53963e951b72883a`. Original Catapult author source audit SHA: `a16eddd34b4339db5ec86e292470ce7929179bc3`.

## 2. Obtain the frozen input datasets

Reproducing **a previous result from archived JSON** and **rerunning a physical SSD experiment** are different levels of reproduction.

- **PubMed1M/MedCPT:** 1,000,000 768-D embeddings; inner product; disjoint first 5K training queries, then held-out 4K warm-up and 1K measured queries. Requires the frozen first-million `.fbin` vectors, the held-out MedRAG/MedCPT query sequence, exact GT and the original DiskANN SSD graph under `$HOME/.cache/geoivf`. Do not substitute another model, vector order, or query distribution.
- **Text2Image-10M:** public first 10M 200-D float vectors, inner-product; first 5K public queries for training, next 4K warm-up / last 1K measured. Public download/preparer is `scripts/prepare_public_ann_dataset.py`. The runner may build the pinned DiskANN index when absent. This can require large bandwidth, RAM and hours of compute.
- **BigANN-10M:** public 10M 128-D `uint8` vectors, squared L2; same disjoint 5K/4K/1K chronology. Public preparer checks official GT and file shape.
- **Coveo chronological production demand:** licensed source with 31,950 normalized 50-D product embeddings, 5K static training, 20K causal warm-up, 5K measured. Original ordering is essential. **Do not distribute the raw licensed records, extracted query text or private manifest paths.** The authorized data must be prepared into the previously frozen local layout. No public download is attempted for this dataset.

Default required local paths and frozen split metadata are **machine-readable** in [`../repro/manifest.json`](../repro/manifest.json), under its `paths` and `controller` keys. Run `navhints doctor --mode <dataset>` for explicit missing filenames. Binary vector files are 8-byte little-endian count/dim headers plus row-major payload (`.fbin` float32; `.u8bin` uint8); GT files store ordered exact IDs and may omit distance payloads. The frozen slicers handle both GT variants, validate row counts and **never query the answer oracle for routing hints**.

Dataset inputs and SSD graph binaries are intentionally **not committed to Git**. A missing input must be acquired legitimately or the exact historical result must be reconstructed from a completed run artifact; substituting approximate data invalidates matched-recall claims.

## 3. Re-run the measured Core and competitors

Inspect commands **without running**:

```bash
navhints run pubmed
navhints run coveo
navhints run catapult
navhints run starling-bigann
```

After verifying free disk space and access to an exclusive NVMe workload window, explicitly authorize the workload:

```bash
navhints run pubmed --execute
navhints run coveo --execute
navhints run text2image --execute
navhints run bigann --execute
navhints run catapult --execute
navhints run memory --execute
navhints run sensitivity --execute
navhints run persistence --execute
navhints run starling-coveo --execute
navhints run starling-bigann --execute
```

If the **public** 10M files are absent, pass `--allow-public-download` explicitly as well. This is not applicable to Coveo. The launcher checks for existing results before writing and refuses accidental overwrite unless `--overwrite` is provided. Each underlying original shell script remains in `scripts/` for full inspection. It writes into `artifacts/` under the checkout. On GitHub, run corresponding `workflow_dispatch` jobs manually on **self-hosted, linux, x64, geoivf**; do not use GitHub-hosted runners or parallel same-device latency measurements.

All experiments use a pinned existing ANN graph with K=10, beam width 8, a measured search-width frontier, online updates only from earlier completed queries, rotated method order, and three repeated replays of the same deterministic query sequence. Do not infer independent-sample standard errors or percentage improvements outside the shared recall interval.

## 4. Fast paper-level reproduction from successful artifacts

The successful original run IDs, artifact IDs/names and required JSON paths are in [the manifest](../repro/manifest.json). To retrieve the original fourteen (ten main/sensitivity campaigns and four service-latency archives) **without rerunning any SSD experiment**:

```bash
gh auth login                      # if needed for GitHub CLI downloads
navhints fetch all                  # dry-run preview; does NOT download
navhints fetch all --execute        # packs each original GH artifact into artifacts/frozen/<id>.zip
navhints verify all                 # check fourteen ZIPs, JSON members and final policy
navhints_paper                      # dry-run figure regeneration and clean PDF build
navhints_paper --execute            # regenerate plots from frozen JSON + make verify-latex
```

To verify a subset: `navhints fetch coveo --execute && navhints verify coveo`. Artifact retention on GitHub is finite. If an original artifact has expired, rerun the **same** causal protocol and archive the new source commit and input hashes before using its output; do not silently fall back to the compact-history control.

The paper commands call:
```bash
cd ../DemmandAwareANN
python3 scripts/derive_final_core_eval.py \
  --coveo ../GeoIVF/artifacts/frozen/coveo.zip \
  --pubmed ../GeoIVF/artifacts/frozen/pubmed.zip \
  --text2image ../GeoIVF/artifacts/frozen/text2image.zip \
  --bigann ../GeoIVF/artifacts/frozen/bigann.zip
python3 scripts/derive_final_core_sensitivity.py \
  --warmup ../GeoIVF/artifacts/frozen/sensitivity.zip \
  --locality ../GeoIVF/artifacts/frozen/sensitivity.zip \
  --shift ../GeoIVF/artifacts/frozen/sensitivity.zip \
  --writes ../GeoIVF/artifacts/frozen/persistence.zip
make verify-latex
```

Use a full local PVLDB/ACM LaTeX template installation and the scripts/Makefile in the paper repository. `make verify-latex` builds into scratch and **does not replace the tracked `main.pdf`**. Check `.latex-verify/result.json`, exact refs, page limits and rendered figure sizes before final publication. Chart CSV regeneration is sourced from measured per-L output and matched-recall comparisons. The presence of a historical PDF file is **not proof** that the active TeX was rebuilt.

## 5. Exact experimental provenance

See [EXPERIMENTS.md](EXPERIMENTS.md). Headline matched-recall SSD read savings on the final four-dataset cohort are 80.69% Coveo, 44.23% PubMed, 34.53% Text2Image-10M and **21.01% BigANN-10M** (the last on four valid common anchors). These are per-dataset means, not the same absolute recall target. The author-aligned Catapult mechanism comparison (run 38127358063) yields 13.28% fewer logical SSD reads for the RAM-only final Core at eleven PubMed targets. Both native Starling replays completed: strong Coveo read and query-latency reductions, but **near break-even and sometimes worse native logical I/O on BigANN-10M**. This is an important negative-boundary result, not a reason to remove BigANN.

All numbers must be interpreted as **logical SSD reads or native Starling logical page counters**, not physical device commands. Auxiliary routing-budget numbers exclude some allocator/hash/table overhead. Sampled persistent writes are logical update requests, not validated crash-consistent storage transactions. Source-aligned Catapult uses its **hash/LRU routing policy within the common DiskANN SSD engine**, not the authors' original complete in-memory search implementation. For these reasons do not claim whole-system native Catapult superiority or universal performance improvement from these experiments.

## 6. Activation failure modes and recovery

- `MISSING file`: check exact split path in `repro/manifest.json`; do not fabricate a different query split.
- `wrong final Core allocation`: refuse artifact and ensure exact B/C/k config or smaller eligible B on Coveo.
- `Cargo build fails`: check the original pinned Microsoft commit and `cargo +1.97.1 --version`; retain the source diff and first compiler error.
- `Starling source patch fails`: source anchors are intentionally strict; pin the original author Starling SHA instead of patching whatever `main` happens to contain.
- `SSH/runners unavailable`: use a genuine local Linux SSD run or artifact-only reproduction. Do not switch to an in-memory ANN engine and call the metric comparable.
- `paper compile failure`: run `make verify-latex` and inspect `.latex-verify/result.json`, not a stale PDF.
- `same NVMe device busy`: finish or stop other jobs before timing. Original scripts provide `flock` device guards; do not interpret overlapping job latency as independent runs.

This workflow is intentionally conservative: it favors visible missing evidence over untestable performance claims.
