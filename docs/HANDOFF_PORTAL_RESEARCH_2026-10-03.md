# Portal research handoff — 2026-10-03

Repository: `gilga1983/GeoIVF`

## Current position

The strongest current result is a **tiny query-specific portal router** in front of otherwise unchanged DiskANN search.

Mechanism:
1. Keep the standard DiskANN/Vamana graph, disk layout, PQ codes, graph traversal, provider I/O, and exact reranking unchanged.
2. Add a tiny RAM routing table: 1024 coarse centroids + one representative DiskANN vertex per cell.
3. For each query, select one representative portal and start ordinary DiskANN from that vertex instead of the global medoid.
4. DiskANN then proceeds normally.

This is attractive because it is almost a retrofit: no graph rebuild policy, no page relayout, no new disk format, and only query initialization changes.

The more ambitious RAM-shadow-graph / mostly-RAM approximate traversal direction should be paused until a focused literature survey establishes a clean gap.

## Important novelty caveat

Do not claim that “using an in-memory routing layer to seed disk ANN” is novel in general.

Closest known work:
- PageANN (arXiv:2509.25487, 2025): page graph plus lightweight/hash-based in-memory routing before disk traversal.
- LAANN (arXiv:2606.02784, June 2026): lightweight in-memory nav graph over page centroids that seeds disk search.
- Amanita Project “Adaptive entry point” experiment (May 22, 2026): publicly proposes query-dependent Vamana entry points from a small auxiliary coarse index.

The defensible question is narrower:

> Can a very small, graph/layout-agnostic, one-vertex router reduce DiskANN I/O and end-to-end latency while leaving the DiskANN index and traversal logic unchanged?

That is the claim to test against prior art.

## Branch / code map

### main
- `main`
- `ff8f54af706bf00b746a97026d0db78bd7007aaf`
- Earlier GeoIVF/PQ/mixed-bit work; no DiskANN portals.

### Seeded DiskANN proof of concept
- `experiment/diskann-seeded`
- tested head `b1f2c0cde4f7af0b6d7896899fb5858fac453b48`
- key files:
  - `scripts/patch_diskann_start_points.py`
  - `scripts/qualify_diskann_seeds.py`
  - `.github/workflows/diskann-seeds.yml`
  - `docs/DISKANN_SEEDED.md`

### Tiny geometric portals
- `experiment/diskann-portals`
- tested run head `e1547dda...`
- key file: `scripts/qualify_diskann_portals.py`

### Cross-dataset suites
- `experiment/portal-suite` — tested `f95e97763372abe5a5c0dca87a5c996864d93c79`
- `experiment/portal-suite-vibe` — tested `d54536fa7e6bdc45944c22018ff1f51dc18ab650`
- `experiment/portal-suite-gist` — tested `f525dfd312ba3704aaff04ddc8a6faee6a7bab04`
- key files:
  - `scripts/fetch_ann_dataset.py`
  - `scripts/qualify_portal_suite.py`
  - `.github/workflows/portal-suite.yml`
  - `.github/workflows/portal-vibe.yml`

### Integrated end-to-end routing
- `experiment/portal-integrated`
- tested `4770955f1593e2bf3db4e4946b6ae6d1e64e17f8`
- key files:
  - `scripts/patch_diskann_integrated_portals.py`
  - `scripts/qualify_portal_integrated.py`
  - `.github/workflows/portal-integrated.yml`

### Current best branch
- `experiment/portal-integrated-fast`
- tested `2071fee70ed5c218f84461ae289f71d254b3a24c`
- **continue from this branch**
- same portal policy, but routing uses DiskANN SIMD `SquaredL2` and a fixed stack shortlist, eliminating per-query heap allocation.

Pinned upstream DiskANN3 revision:
`fcf90534174cf29c78c9f13b4cccf1fcabff85f5`

## What the patch does not change

Portal experiments leave unchanged:
- Vamana graph construction and topology;
- disk layout;
- PQ data;
- graph traversal after initialization;
- exact full-vector reranking;
- provider I/O policy.

That narrow scope is one of the main advantages.

## Protocol

Integrated VIBE runs:
- node cache disabled;
- 1 search thread;
- beam 8;
- `L in {60,100,200,400}`;
- one portal per cell;
- `nprobe in {1,8,32}`;
- 1024 cells;
- 256 deterministic held-out queries;
- 3 repetitions;
- Linux DiskANN uses `O_DIRECT`.

Datasets successfully evaluated:
- SIFT1M 128D L2
- GloVe-200 angular
- GIST1M 960D L2
- Yahoo MiniLM-384 normalized
- ImageNet CLIP-512 normalized
- COCO Nomic-768 normalized, OOD text-to-image

NYTimes contains zero-norm vectors and was intentionally not silently normalized.
Fashion-MNIST did not run because the classic sequential workflow stopped at NYTimes.

## Key results

### IVF-PQ seed proof of concept, SIFT
Fresh 256 held-out queries, DiskANN L60/beam8:
- medoid: 85.418 I/Os/query, 3541.1 comparisons, Recall@10 99.1797%
- IVF-PQ np2-r1: 62.352 I/Os/query, 2323.2 comparisons, same recall
- **27.0% fewer I/Os, 34.4% fewer comparisons**

### Tiny portals, SIFT
- medoid ~85.15 I/Os/query
- ~1 MiB single-portal table: ~69.13 I/Os/query
- deeper routing with same ~1 MiB table: ~68.01 I/Os/query
- 69 MiB IVF-PQ seeder: ~62.62 I/Os/query

Most seed benefit survives with tiny RAM overhead.

### Optimized integrated router
Workflow `37010905319`
Artifact `portal-integrated-37010905319`
Commit `2071fee70ed5c218f84461ae289f71d254b3a24c`

Routing overhead after optimization:
- Yahoo MiniLM: ~20.5–31.0 us/query
- ImageNet CLIP: ~25.4–37.0 us/query
- COCO Nomic: ~39.5–51.6 us/query

#### L=100
Yahoo MiniLM:
- medoid: 2600.13 us, Recall 99.6094%, 123.64 I/Os
- portal np1: **2251.39 us**, Recall 99.5313%, 105.78 I/Os
- **13.4% lower latency, 14.4% fewer I/Os**

ImageNet CLIP:
- medoid: 2784.89 us, Recall 99.2578%, 128.75 I/Os
- portal np1: **2244.28 us**, identical recall, 106.17 I/Os
- **19.4% lower latency, 17.5% fewer I/Os**

COCO Nomic OOD:
- medoid: 3277.87 us, Recall 12.8516%, 147.94 I/Os
- portal np32: **2851.73 us**, Recall **14.6875%**, 123.19 I/Os
- **13.0% lower latency, 16.7% fewer I/Os, higher recall**

#### L=200
Yahoo:
- 4530.03 us -> **4221.10 us**, identical 99.8438% recall
- **6.8% faster**

ImageNet:
- 4735.00 us -> **4212.09 us**, identical 99.8047% recall
- **11.0% faster**

COCO np32:
- 5329.30 us -> **4914.80 us**
- recall 20.7422% -> **21.9531%**
- **7.8% faster**

### GIST1M
Workflow `37001898581`, artifact `portal-gist-37001898581`

L100:
- medoid 127.17 I/Os, Recall 82.1875%
- portal np8 110.38 I/Os, Recall 81.7578%
- **13.2% fewer I/Os**

L400:
- medoid 421.40 I/Os, Recall 96.6406%
- portal np8 405.66 I/Os, identical recall
- **3.74% fewer I/Os**

Treat I/O/recall as primary for GIST unless rerun with the integrated router.

## Interpretation

The portal result now generalizes across classic, semantic, high-dimensional, image, and OOD workloads.

The strongest positioning is not “we invented query-specific entry points.” It is:

- a **minimal retrofit** to DiskANN;
- very small extra RAM;
- unchanged graph/layout/search logic after initialization;
- consistent I/O reduction;
- true end-to-end latency reduction once routing is implemented efficiently.

Approximate RAM-navigation / shadow-graph ideas remain interesting, but have much heavier PageANN/LAANN/two-tier prior-art overlap.

## Next task

Start the next chat with a **focused literature survey**, before more coding.

Questions:
1. Has anyone added a tiny query-dependent entry router to otherwise unmodified DiskANN/Vamana, with no page graph, no graph rebuild, no nav graph, and one selected start vertex?
2. How exactly do PageANN hash routing and LAANN nav-graph routing differ from this portal in memory, build requirements, number of seeds, and index/layout changes?
3. What exactly did the May-2026 Amanita adaptive-entry experiment implement and measure?
4. Are there peer-reviewed systems using IVF/coarse quantization only to choose a Vamana/DiskANN entry point?
5. Is the contribution best framed as portal routing, a retrofit optimization, entry-point distillation, or a memory/latency operating point?
6. For the ambitious RAM-navigation idea, identify the closest prior systems before implementing it.

After the survey, choose among:
- keep portal as a standalone component/contribution;
- learn better portals at the same tiny footprint;
- jointly reduce DiskANN L at matched recall;
- or pursue RAM navigation + SSD verification.

## Workflow / artifact IDs

- seeded DiskANN: run `36980809665`, artifact `11216170283`
- tiny portal: run `36984265070`, artifact `11217261224`
- VIBE suite: run `36991949488`, artifact `11221422521`
- GIST1M: run `37001898581`, artifact `11225404538`
- first integrated router: run `37005199210`, artifact `11226582846`
- optimized integrated router: run `37010905319`, artifact `11229830123`

## Test status

Integrated experiments repeatedly passed:
- 327 tests passed
- 0 failures/errors
- 18 optional tests skipped

## Cautions

- Use integrated in-process latency from `portal-integrated-fast`, not old Python/Faiss composed latency.
- DiskANN `mean_ios` is provider-level I/O/vertex-load accounting, not independently counted NVMe commands.
- COCO OOD absolute recall is low under the fixed DiskANN configuration; use it as same-configuration comparative evidence.
- The broad entry-point concept has nearby prior art, so novelty must be stated narrowly and supported by the literature survey.
