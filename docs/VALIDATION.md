# Evaluation-foundation validation

Date: 2026-09-29

Tested source commit: `c8e884db3d8d4e8f931d02bf38b65665c3dc35c1`

Successful GitHub Actions run:
https://github.com/gilga1983/GeoIVF/actions/runs/36520790337

## Executed checks

The self-hosted integration job completed successfully on the GeoIVF runner03
installation. It installed Faiss 1.15.1 and NumPy 2.5.3 in an isolated virtual
environment, built the synchronous native reader, and ran the tests.

- Unit/integration suite: 31 passed, 1 skipped. The skipped test is the optional
  io_uring backend test. Native O_DIRECT reads and Faiss equivalence were tested.
- Local isolated-container suite: 30 passed, 2 skipped. Faiss was unavailable
  locally, in addition to the untested io_uring path.
- Synthetic integration fixture: 4,096 FP32 128-dimensional base vectors,
  32 queries, 16 IVF lists, nprobe=8, k=10.
- Three layouts (input, radial, geometric) each ran with no filtering and with
  combined conservative filtering. All 32 queries in each of the six
  configurations matched Faiss IVFFlat's returned neighbor IDs: 192 successful
  configuration-query comparisons, not 192 independent queries.
- Pinned upstream MQSim commit
  `51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16` compiled and replayed an exported trace.
- Independent inspection of its archived XML verified 105 generated READ
  requests, zero WRITEs, and 4,272,128 transferred read bytes, exactly matching
  the exported request count and byte sum.

Artifact: `geoivf-integration-36520790337`, GitHub artifact ID `11012791377`.
Archive SHA256:
`3325089e9e5b8ecbf830202f42633daf546dc9eb60f5c4d2371ec2b2b855dc74`

The artifact contains the JUnit report, resolved package versions, host and
commit information, six dependency-bearing JSONL traces, and the MQSim trace,
configuration, console log, and result XML. GitHub artifact retention is 14 days;
retain a durable copy for the research record.

## What this does not establish

These are implementation/integration checks, not evidence of speedup on real ANN
benchmarks. The fixture is synthetic, small, and deliberately easy to validate.
MQSim uses a small integration-only device configuration and synthetic fixed
request arrivals. It does not enforce the JSONL query-stage dependencies, so
its result is not an end-to-end ANN latency or QPS measurement.

CLIP/HIVF-CLIP are not integrated. The current radial bound is deterministic
reverse-triangle pruning, not the CLIP learned bound. The current projection is
coordinate selection, not PCA. Radii occupy four bytes, not one. The planner and
exact-distance loop remain in Python/NumPy; the optional liburing path still
needs compilation and execution validation. Canonical datasets and controlled
NVMe performance experiments remain the next validation gates.
