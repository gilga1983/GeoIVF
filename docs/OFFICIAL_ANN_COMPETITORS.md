# NavHints: strong published disk-ANN competitors

Status: official-code comparison in progress. This is **separate** from the
controlled routing/caching ablations. The paper's approved results have not
been rewritten and **no unmeasured SOTA comparison is claimed**.

## Why

A native 30-node BFS cache or workload-trained full-node cache is a valid
equal-*auxiliary-RAM* control, but not a published whole-system competitor.
QSEV-32 is only the query-sensitive entry-selection component of DiskANN++;
it omits DiskANN++'s isomorphic disk remapping and asynchronous PageSearch.
These controls belong under *mechanism ablations*, not a SOTA headline.

Our CatapultDB experiment implements its mechanism in the pinned Microsoft
DiskANN benchmark. It is an **independent implementation**, not verified as
an author-released binary. A public official code repository has not yet
been confirmed. CatapultDB remains the closest **algorithmic** related work;
compare paper defaults and large-RAM settings, validate the same causal
history, and do not call a port the author's official implementation.

## First full-system reproduction targets

| System | Original source | Pinned revision | What it includes |
| --- | --- | --- | --- |
| Starling (2024) | https://github.com/zilliztech/starling | `17dc3e8a011533a62374445f53963e951b72883a` | In-memory navigation graph, layout partition, page search, graph cache |
| GPS/Gorgeous (2026) | https://github.com/yinpeiqi/Gorgeous | `04c8c27d77a19e7751748d4e0ac4ecd2f0fb3e8b` | Graph-priority caching, graph-replicated disk layout, in-memory navigator, asynchronous search |

The GPS publication cites an implementation whose GitHub project is named
Gorgeous. Preserve its actual authors' system terminology rather than
misrepresenting it as identical to an arbitrary DiskANN patch.

Phase 1: reproduce a **clean build and native search CLI** from each pinned,
unmodified author repository on a self-hosted machine, without running any
SSD benchmarks in parallel with existing tests. Check that the same
BigANN-10M base vectors and 5,000-request held-out set are available, with
the exact official ground truth and original vector data types. The smoke
workflow only checks prerequisites and native binaries, no performance.

Phase 2: on the common public BigANN-10M dataset (128D uint8, squared L2),
build **each system's complete recommended index** including its auxiliary
navigator and disk layout. The systems may rebuild indices using their
native implementations: record construction time, degree, PQ bytes, graph
disk footprint, auxiliary RAM and observed process RSS.

Phase 3: benchmark **the same query IDs and official GT** under
4 threads, Recall@10 and a sufficient full recall frontier. Sweep native
search-list size, navigation memory, and beam width as authors recommend.
Report hardware NVMe median/mean latency, QPS, read operations/bytes when
available, p95/99 where available, peak/steady RSS, and index size. Use the
shared `$HOME/.cache/geoivf/speed-device.lock` *before* measurements;
rotate run order and take three repetitions. Report the authors' own
baseline too, because different codebases, compilers and index
construction may otherwise distort a cross-system speed comparison.
Never infer SSD reads from requested graph expansions.

Phase 4: repeat on Text2Image-10M (200D float32 inner product) if both
systems natively support the metric, and consider a third workload (Coveo
or PubMed) only where their native dimension/data path supports it.
Do **not** silently convert cosine/IP to L2, use different GT,
silently change the query split, or compare different recall ranges.

Starling's official README documents BigANN uint8/L2 and Text2Image
float32/IP compatibility. Its native full configuration includes a memory
graph and partitioned page search, not merely beam search.
Gorgeous' code and documentation explicitly list its graph-priority
cache, graph-replicated layout, memory navigator and pipelined search.
The eventual strong comparison must enable those features, or be clearly
marked as a partial configuration.

## Additional systems and limitations

- **DiskANN++** (arXiv 2310.00402): a QSEV-only implementation is not
  whole-system DiskANN++. Seek author source or a faithful full reproduction
  before making full-System comparisons.
- **OctopusANN** (PVLDB 2026): benchmark umbrella and Starling/PipeANN
  forks: https://github.com/LeonLee666/IObench4DiskANN. Investigate once
  the official Starling/Gorgeous runs work.
- **PipeANN**: https://github.com/LeonLee666/PipeANN; comparable for
  pipeline throughput and latency, although it addresses somewhat
  different optimizations.
- **AiSAQ**: https://github.com/kioxia-jp/aisaq-diskann; different
  all-in-storage memory contract, useful only as a clearly labeled
  separate memory tradeoff.
- **CatapultDB**: the closest query-locality mechanism, with currently
  no verified author-release source. Label our implementation accurately
  and report its choice of hash count, bucket size, memory, and training.
- **30-node native/workload-hot cache**: keep as a mechanism-cost control,
  never call it state of the art.

## Decision rule

The true systems comparison should show whether NavHints, with its
~103 KB auxiliary payload, is competitive with systems using their
recommended (often much larger) in-memory navigation structures and
index layouts. If they outperform NavHints, report that and emphasize
different resource budgets and composability. Do not choose a weak
configuration because it makes NavHints win.

Only promote a method to the main related-work results table after
validating its full feature path and identical workload/recall contract.


## Native integration finding: Gorgeous replica-layout input sector count (2026-10-09)

The original Starling and Gorgeous projects both compile on the self-hosted runner
after provisioning their Ubuntu development dependencies and setting
`CMAKE_POLICY_VERSION_MINIMUM=3.5` for their bundled CMake/oneTBB versions.

The original Starling implementation completed a native 20K-vector/200-query
BigANN pilot with Recall@10 measured. These tiny-pilot numbers are **not**
10M dataset benchmark results.

The pinned Gorgeous author revision `04c8c27d77a19e7751748d4e0ac4ecd2f0fb3e8b`
segfaulted in `tests/utils/index_relayout_free_mem.cpp` during its
graph-replicated `gr_layout` construction. The diagnostic
[run #37928612627](https://github.com/gilga1983/GeoIVF/actions/runs/37928612627)
(artifact `official-native-pilot-37928612627`) preserved:

- Upstream input index: 20,000 records and **10** records/sector.
- Replica partition map: 20,000 valid partitions and capacity **15**,
  verified with no empty partitions, mismatched first IDs, over-capacity
  partitions, or invalid vertex IDs.
- Relayout allocated/read **1,334 input sectors**, using replica `C=15`,
  although the original disk-index layout requires **2,000 input sectors**,
  using `nnodes_per_sector=10`.
- GDB confirmed a SIGSEGV in `memcpy` while processing graph replicas.

The integration pilot now applies an **exact, one-line correctness fix** in
the temporary checked-out author source, and archives the resulting source
diff in `gorgeous-relayout-sector-count-fix.diff`:

```diff
-  auto diskann_partition_number = ROUND_UP(_nd, C) / C;
+  auto diskann_partition_number = ROUND_UP(_nd, nnodes_per_sector) / nnodes_per_sector;
```

This changes how many **source index sectors** are loaded, without changing
the graph, partition assignment, search logic, or query routing decisions.
The patch must be **disclosed** with any eventual Gorgeous performance
comparison: the benchmark would use the pinned *original author system with
a one-line input-buffer correctness fix*, not entirely unmodified author
code. A new native diagnostic run is
[#37929860156](https://github.com/gilga1983/GeoIVF/actions/runs/37929860156).

The pilot remains a qualification gate. Do not infer end-to-end 10M
performance, a systems winner, or a full-systems SOTA comparison from it.
