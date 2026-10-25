# Reading and maintaining NavHints

NavHints adds workload-trained junctions and causal recent destinations to
existing disk-resident ANN engines. Training, engine integration and numerical
review are separate parts of the implementation.

## Start with the path you need

| Task | Entry point | What to follow next |
| --- | --- | --- |
| Check the paper's recorded results | `repro/run.py review` and `repro/verify_review_bundle.py` | Source manifests, aggregate checksums, matched-recall comparisons |
| Regenerate plots | `repro/export_paper_results.py` | Named export functions, then the pinned derivations in `repro/paper/scripts/` |
| Train the junction directory | `scripts/learn_navhints_generic.py` or `scripts/learn_global_navigation_landmarks.py` | `scripts/build_hint_ivf_generic.py` or the PubMed builder |
| Inspect the deployed DiskANN routing | The three patch layers below | Provider selection, causal benchmark feedback, explicit replay configuration |
| Reproduce an SSD experiment | `repro/run.py list` and `scripts/run_final_core_*.sh` | Frozen input paths, pinned engine revision, measured replay harness |
| Inspect the native Starling port | `scripts/patch_starling_final_core.py` | `page()` supplies starts; `benchmark()` updates the unique FIFO history |

## The DiskANN patch layers

The replay scripts apply these layers to an isolated checkout of the pinned
upstream engine, in this order:

1. `patch_diskann_hint_ivf_packed_direct.py` installs the compact learned
   junction selector. Its provider function scores region representatives
   and candidate IDs with the engine's resident PQ lookup table.
2. `patch_diskann_experience_core.py` installs successful-destination feedback
   and the optional persistent-exit machinery. Its provider and benchmark
   functions describe the state crossing the search boundary.
3. `patch_navhints_regional_history.py` supplies history selection and multiple
   PQ-ranked starts. The paper replay explicitly chooses global unique FIFO,
   capacity 2,048 and up to eight additional starts. Other modes support
   controlled comparisons.

The Python scripts contain exact upstream source anchors and replacement
Rust blocks. Read the replacement blocks to inspect the engine algorithm;
read the surrounding Python to see the source-version checks and insertion
points. A replacement must find the expected source shape exactly. Editing
an anchor as though it were a prose comment can break patch application.

The graph, bounded frontier and stopping rule remain with the upstream
engine. Feedback is applied after a completed query. Repeated destination
IDs do not refresh the selected FIFO policy's age.

## Names retained by the experiment format

Some short names are also keys in archived JSON, environment variables or
upstream logs. They are compatibility boundaries, not alternative designs.

| Identifier | Meaning |
| --- | --- |
| `baseline` | Unmodified engine search |
| `ivf` | Learned junction-only routing |
| `core` | RAM-only NavHints with recent destinations |
| `full` | Core plus the separately evaluated sampled page exits |
| `cache` or `value_cache` in the routing patch | Destination IDs, rather than cached graph records or reusable query answers |
| `hub` in the routing patch | A selected learned junction |
| `L`, or `search_width` in Python loops | The engine's search-list capacity |

Keep these serialized identifiers stable when improving local Python names.
Likewise, preserve the documented recall grids, interpolation rules and
per-repetition/seed averaging order. The complete-request latency analysis
has a different target grid from the headline read-saving analysis.

## Shared input preparation

`scripts/navhints_io.py` owns vector and ground-truth slicing for the two main
replay harnesses. Headers contain little-endian row/dimension counts. Ground
truth stores IDs and optional distances in separate blocks. Coveo requires
both blocks; other public workloads can supply IDs alone.

Slices are written to a temporary sibling file and installed after a complete
copy. Invalid ranges, truncated payloads and source/output aliases fail with
the input path in the error. The harnesses retain small compatibility wrappers
because other paper experiments import those helper names.

## Starling source generation

The final Starling generator composes strict transformations of a shared
native replay template. Several helper filenames refer to intermediate
controls because their source transformations are still used. The four final
generated shell scripts are checked against frozen pre-cleanup hashes in
`repro/starling-replay-reference.json`. Generation and syntax checks require
no ANN dataset or physical SSD replay.

## Validate a change

```bash
make test
make review
make paper-data
```

The tests include binary-layout failure cases, source dependencies, generated
Starling replay hashes and malformed plot CSV diagnostics. Numerical review
recomputes the recorded results. Plot export checks all 20 manuscript CSVs.
Physical latency measurement remains a separate, explicitly executed task.

Contributor formatting and static checks are optional dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
make format-check
make lint
```

`make format` applies the configured style to maintained Python. The frozen
`repro/paper/` snapshot is excluded: its exact source bytes are part of the
artifact provenance and should change only with a deliberate paper release.
Use function names, type annotations and short explanations of invariants to
make changes readable; keep runtime source strings and evidence schemas
stable unless the scientific change explicitly requires otherwise.

The formatting-only commit is recorded in `.git-blame-ignore-revs`. To follow
the original authorship locally, use `git blame --ignore-revs-file
.git-blame-ignore-revs <path>`.
