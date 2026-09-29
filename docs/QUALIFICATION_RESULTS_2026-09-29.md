# Canonical qualification results, 2026-09-29

## Executed scope

Implementation commit: `559dd2be808c2a9587abdc7ae136ba0e6fff2a10`.
Private BLAS/LAPACK build fix: `3098cdef45fbee094fcd15efc9c8558ab9af48d6`.

First qualification run:
https://github.com/gilga1983/GeoIVF/actions/runs/36522607437

Repeat qualification and successful upstream build:
https://github.com/gilga1983/GeoIVF/actions/runs/36522828308

Both jobs in the second run completed successfully. All 40 tests passed, with
no skipped tests, including native O_DIRECT, io_uring at depths 1/2/8/32,
unique-page buffer association, short-read/error propagation, and full-search
result equivalence. Upstream CLIP, HIVF-CLIP and IVFFlat executables compiled.
Their search/training code and bundled Faiss revision were not modified.
The first CLIP build failed because host BLAS was unavailable. The fixed setup
privately downloads/extracts Ubuntu BLAS/LAPACK packages; it does not use sudo
or install system packages. Package versions and hashes are in the artifact.
Compiled CLIP has NOT yet been benchmarked on SIFT1M or instrumented for I/O.

## Dataset and methodology

The canonical TexMex ANN_SIFT1M archive was downloaded from:
ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

Archive SHA256:
`92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`

Base-file SHA256:
`21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`

Validated 1,000,000 base vectors, 10,000 query vectors, 128 FP32 coordinates,
10,000 ground-truth rows of 100 neighbors, and the 100,000-vector learning file.
These are observed integrity hashes, not upstream-signed checksums.

Configuration: IVF nlist=1024; construction seed=12345; a fixed 100,000-vector
base sample for Faiss training; k=10; nprobe=4/16/64; 4 KiB FP32 payload pages;
8 vectors/page; 16 independent development queries. Query IDs:
6822, 4727, 98, 6937, 7585, 5610, 3758, 8491, 5401, 549, 1997, 4885, 9265,
361, 6311, 3798.

Every method received identical preassigned IVF candidates. The candidate-route
SHA256 was `248c5e8bbd83b8173081d4a2bf3dbc21f2345134c0550aedfc72ecdff5a3b15b`.

The fixed-representation comparison crossed three layouts (input, radial,
GeoPack) with four filters (none, deterministic radial, balls, combined), using
8 balls and 16 variance-ranked original coordinates. A further sweep used
4/8-coordinate summaries on exactly the same frozen GeoPack pages.
This is 48 configurations including the nprobe settings.

The GeoPack payload SHA256 was identical for all three summary dimensions:
`8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7`.

## Correctness

All 768 configuration-query comparisons matched independent FP64 exhaustive
search over the same selected IVF lists, with ID tie-breaking. This is 16
independent queries, NOT 768 independent queries.

For the first two queries in each configuration, 610 rejected-page decisions
were independently audited after the search. Every audited page had a true
minimum distance greater than the threshold at rejection, and the lower bound
was conservative. These decisions may refer to the same pages across different
configurations; this is not a count of distinct pages.

Two full-database checks also agreed with the canonical top-10 ground truth
in both distances and IDs. The repeat run reproduced all 48 configurations'
recall, correctness, candidate counts, requested pages/bytes/extents/stages and
distance-evaluation counts exactly. This is reproducibility, not an additional
independent query sample. Per-query CSV sums were independently reconciled with
the aggregate JSON and with read_bytes = 4096 * read_pages.

## Main result: current summaries are not selective enough

These are requested page/extent counts from MemoryReplay of actual FP32 payloads.
They are NOT observed SSD traffic or NVMe timings.

| nprobe | Recall@10 | Unfiltered pages/query | GeoPack + 16D combined pages/query | Page reduction |
|---:|---:|---:|---:|---:|
| 4 | 0.74375 | 580.5000 | 579.8125 | 0.1184% |
| 16 | 0.93750 | 2207.6250 | 2197.7500 | 0.4473% |
| 64 | 0.99375 | 8448.9375 | 8284.8750 | 1.9418% |

At nprobe=64 with the same 16D combined representation:

| Layout | Requested pages/query | Requested extents/query |
|---|---:|---:|
| Input | 8381.1250 | 195.0625 |
| Radial | 8319.5000 | 189.3750 |
| GeoPack | 8284.8750 | 205.1250 |
| Unfiltered, any layout | 8448.9375 | 65.0625 |

GeoPack gives modest additional page pruning but more extents in this setup.
The unfiltered baseline uses whole-list planning; filtered search uses 64-page
windows. Therefore the extra requests reflect BOTH stage boundaries and
fragmentation. They must not all be attributed to the page layout. A same-window
unfiltered control and a scheduling/coalescing sweep remain necessary.

Savings are highly concentrated: at nprobe=64, nine of sixteen queries save no
pages. Query 98 accounts for 2,455 of the total 2,625 saved pages. Median per-query
page reduction is zero. The small development sample is not enough for a final
performance conclusion, but it does not support a broad speedup claim.

For the same GeoPack payload and combined filtering at nprobe=64:

| Summary dimension | Geometry bytes/page | Plus radial bytes/page | Pages/query |
|---:|---:|---:|---:|
| 4 | 64 | 72 | 8441.8750 |
| 8 | 96 | 104 | 8404.8750 |
| 16 | 160 | 168 | 8284.8750 |

There are 125,445 pages. The rich geometric summary occupies 20,071,200 bytes;
radial intervals occupy another 1,003,560 bytes. Complete allocated directory
arrays occupy 29,895,122 bytes (about 28.51 MiB). Python runtime and decoded
transients are extra. No-filter runs currently load the same directory arrays;
this is not a minimal-RAM baseline implementation.

## Interpretation and remaining work

The infrastructure and correctness gate passed. The current 16-original-coordinate
representation did not pass a useful-pruning gate on this development sample.
It is not PCA, and these results do not reproduce the large reductions reported
in earlier small exploratory experiments. Those earlier percentages must not be
used as evidence for this canonical implementation.

No end-to-end speedup, CLIP comparison, or calibrated-device result is claimed.
The next useful investigation is bound quality on fixed pages, including PCA
versus coordinate selection and stronger compact quantized representations,
followed by the window/coalescing controls. More hardware benchmarking alone
will not strengthen weak bounds. Upstream CLIP is now available for the actual
canonical-data integration rather than a reconstruction.

## Durable artifact identifiers

First canonical artifact: `11013553195`, SHA256
`20ba5d80b9c32d884a0a242df69de5d633de176c0dce52d8b331664987b6c2b7`.

Repeat canonical artifact: `11013014764`, SHA256
`96c02ac9769026129a62cd1962c5f5667900853147614c6b5162769300140c13`.

Successful upstream/tests artifact: `11013523546`, SHA256
`757a695b26c7eeb25b3ad660d327758e6d297a20b871e35283ead29d29ba6b99`.

Artifacts include per-query CSVs, full result JSON, query IDs and shared routes,
source commit, package versions, dataset manifest, JUnit report and build logs.
GitHub retention is 30 days; downloaded copies should be retained for the paper.
