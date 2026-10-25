# NavHints reviewer release (25 October 2026)

The public entry point is https://github.com/gilga1983/GeoIVF.
Reviewer numerical checks and standalone plot regeneration work from this
checkout without the private manuscript repository or expiring Actions files.

## Completed

- Bundled 48 aggregate JSON outputs from 35 original experiment archives.
- Recorded exact evidence ZIP SHA-256 in `repro/release-20261025.json`.
- Recovered both Table 5 originals; validated their original ZIP digests and
  source provenance. Independent recall/read interpolation reproduces all
  three source controls and six history-capacity results as printed.
- Extended the source inventory to 32 required experiments; three additional
  source archives remain optional controls.
- Recomputed headline read savings, complete-request latency comparisons,
  Figure 8 and four-workload junction scoring from checksummed aggregates.
- Added the frozen original plotting sources and derivation scripts from
  manuscript commit `0af8904f3885979a85af59bc5bd732ba9ee3c087`.
- Regenerated all 20 plotted CSVs and checked numerical agreement with the
  current manuscript reference files.
- Added a standalone eight-figure/nine-table TeX sheet using those sources.
- Added explicit permission to review and reproduce original contributions.
  This is not a general open-source license; third-party terms remain separate.
- Preserved both newly recovered original Table 5 archives separately from
  the public aggregate evidence.

## Scope

This release supports source inspection, recorded-result verification and
plot regeneration. It is not a new physical SSD replay. Exact raw datasets,
query text, graph/index binaries, per-request profiler logs and host diagnostics
are excluded. Full experiment scripts require the original authorized inputs
and pinned engines, and a suitable isolated NVMe machine.

The plotted CSVs are recomputed. The TeX tables are frozen manuscript outputs;
the independently checked numerical coverage is listed in ARTIFACT_MAP.md.
No claim is made that every table is independently recalculated by the
lightweight verifier. No DOI or separate artifact badge is claimed.
