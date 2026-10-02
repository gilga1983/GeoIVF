# GIST1M portal stress rerun

This branch carries the same fixed portal policy and common DiskANN L sweep used by
the cross-dataset portal study. The previous GIST workflow was cancelled only because
GitHub Actions replaced a pending run in the shared performance concurrency group.

Policy is unchanged:
- one geometric portal per IVF cell
- portal routing nprobe in {1, 8, 32}
- DiskANN beam 8
- common L sweep {60, 100, 200, 400}
- no per-dataset tuning

This commit exists to enqueue the stress test after the shared performance slot reopened.
