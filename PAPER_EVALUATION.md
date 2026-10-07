# NavHints final paper evaluation contract

This branch freezes the evaluation contract for the PVLDB/VLDB paper.

## Frozen controller

Canonical controller patch blob:

`scripts/patch_diskann_experience_core.py`
SHA: `2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb`

Core:
- learned 16K ID-only Hint-IVF entry directory;
- 512 unique recent-result IDs, SkipDup/FIFO;
- optional persistent 10-ID hub exits;
- stateless Sample2 persistence for the default persistent configuration:
  each eligible novel hub winner is admitted with probability 1/2;
- one-slot FIFO replacement on every admitted persistent update;
- ranked cache candidates may fill empty persistent slots but never cause eviction;
- no support-2, regional routing, stage maps, or augmented associated-data record format.

Persistence is optional by design. The core entry+cache configuration must always be reported separately.

## Search contract

Unless a specific sensitivity experiment says otherwise:
- DiskANN pinned revision: `fcf90534174cf29c78c9f13b4cccf1fcabff85f5`;
- K = 10;
- beam width = 8;
- 4 search threads;
- same graph, PQ state, index file, SSD and process epoch for compared methods;
- same configured search-list budget L within a raw run;
- conclusions are made at matched recall by interpolation, not same-L alone;
- method order is rotated across repetitions;
- timing includes routing/PQ scoring in the query path;
- online structures begin empty unless the experiment explicitly studies steady state.

## State accounting

Report separately:
1. static 16K routing bytes;
2. 512 cache IDs;
3. transient/runtime container overhead where measured;
4. persistent hub IDs, including whether they fit existing page slack;
5. graph-file growth and sectors/read if any;
6. logical page writes/query and physical write amplification where available.

Competitor accounting must not charge NavHints container overhead while silently omitting analogous competitor overhead. If anything is excluded, state the exclusion and bias it in favor of the competitor.

## Primary configurations

- DiskANN baseline.
- Learned 16K entry only.
- Core NavHints = 16K entry + 512 recent-result cache.
- Full NavHints = Core + optional Sample2 persistent hub exits.
- Native/equal-budget node cache where meaningful.
- DiskANN++-style QSEV.
- CatapultDB best reproducible configuration.
- Starling if a faithful same-device comparison remains tractable.

## Required evaluation matrix

### A. End-to-end PubMed/MedCPT
Matched-recall I/O and latency for all primary configurations.

### B. Public 10M datasets
- Text2Image-10M, inner product.
- BigANN-10M, squared L2.
Report core and persistent increments separately.

### C. Persistence sampling frontier
Sample1 / Sample2 / Sample4 / Sample10:
- matched-recall I/O and latency;
- admission fraction;
- logical writes/query;
- FIFO evictions;
- page occupancy.

### D. Warm-up
From empty online state, plot benefit versus completed requests. Core cache and optional persistent layer should be separable.

### E. Locality
Vary temporal/query locality while preserving the same static entry structure. Report where the recent-result cache and persistent layer stop paying.

### F. Distribution shift
Train the static entry map on one window/population, evaluate on a shifted window/population, and show online adaptation.

### G. Scale
At least 10M is mandatory and already supported. Attempt 100M only if it does not displace higher-value paper gaps.

### H. Systems accounting
- serialized and RSS state;
- cache scan CPU;
- graph sectors/read;
- graph-file growth;
- logical and physical write costs for persistence.

## Evidence hygiene

Development experiments are not submission results unless they obey this contract. Old progressive-routing, support-2, regional, and saturating-Fill results may appear only as historical design/ablation evidence with their protocol stated explicitly.

Do not change the frozen controller while filling the evaluation matrix. If a run exposes an implementation bug, fix the bug without changing semantics and record the new canonical patch blob here.
