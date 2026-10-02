# DiskANN as a carrier for RAM IVF-PQ graph seeding

## Question

Can a compact RAM IVF-PQ index provide better graph entry vertices and thereby reduce
DiskANN's own disk-backed graph traversal work, without changing Vamana, the on-disk
graph/vector records, DiskANN's PQ representation, or its full-precision reranking?

This experiment isolates navigation. GeoPack and certified page completion are deliberately
excluded until the seed effect is established.

## Carrier modification

Pin Microsoft DiskANN revision `fcf90534174cf29c78c9f13b4cccf1fcabff85f5`.
A fail-closed patch adds a public search method that supplies query-specific starting
vertex IDs to the existing disk SearchAccessor. With no supplied IDs the released medoid
behavior is unchanged. The same Knn traversal, disk provider, I/O counters, PQ distance
code, graph, exact-distance cache, reranking, beam width, and disk files are used.

The benchmark accepts an optional fixed-width u32 seed matrix through
`DISKANN_START_POINTS_FILE`. This is harness plumbing, not an alternative graph search.

## RAM seeder

Use a separate one-thread Faiss IndexIVFPQ over the same million SIFT vectors:

- nlist = 1024
- 64 PQ chunks, 8 bits/chunk = 64 code bytes/vector
- train on the separate SIFT learning set
- original IDs preserved

Sweep nprobe in {1,2,4,8,16} and seed count in {1,2,4,8}. Query seeding is timed one
query at a time. This pilot duplicates a roughly 64-byte/vector compressed code store
plus IVF IDs; it does not yet reuse DiskANN's already-resident PQ codes. That extra RAM
must be charged.

## Search protocol

Keep DiskANN at L=60 and beam=8, the previously selected released-baseline operating
point. Node caching remains disabled. Therefore development tunes only the seed policy,
not DiskANN's search breadth.

Development queries are permutation(seed=20260929)[2024:2152].
Held-out queries are permutation[2152:2408]. Both exclude every cohort used by the
earlier GeoIVF representation experiments.

Development selects three diagnostic policies among configurations achieving at least
99% upstream Recall@10:

1. minimum DiskANN mean I/O counter,
2. minimum DiskANN internal mean latency,
3. minimum composed diagnostic latency = separately measured Faiss seed time +
   DiskANN internal mean latency.

Those choices are frozen before held-out evaluation. Held-out runs include the unchanged
medoid baseline and the distinct selected seed policies, three rounds each.

## Metrics and interpretation

Primary metrics are upstream DiskANN mean I/O operations, graph hops, PQ comparisons,
I/O time, search latency, and recall. Because all arms use the same provider counter,
relative changes in mean I/O are meaningful as DiskANN vertex-load work. The counter is
not claimed to equal physical 4 KiB reads or NVMe commands.

Seed selection time is reported separately. Adding it to DiskANN's internal timer is a
diagnostic composition, not an integrated production latency measurement because the
Faiss seeder and DiskANN benchmark currently run in separate processes.

A useful outcome is a recall-preserving reduction in DiskANN graph I/O. Only after that
is established should GeoPack page placement and our certified completion layer be added.
