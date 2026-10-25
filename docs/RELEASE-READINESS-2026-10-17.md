> Historical audit. Superseded by [the 25 October release status](RELEASE-READINESS-2026-10-25.md).

# NavHints artifact release status (October 17, 2026)

This public repository is the code and reproducibility entry point for the
NavHints regular research paper. Full experiment archives have not yet been
deposited in a permanent public repository.

## Reproduction entry points

- [Artifact instructions](ARTIFACT_REVIEW.md)
- [The 14-original-run provenance manifest](../repro/manifest.json)
- [Approved supplemental results manifest](../repro/supplemental-manifest-20261014.json)
- [Warm-up across starting-point counts](../scripts/qualify_final_core_warmup_starts.py)
- [Alternative junction scoring](../scripts/qualify_junction_score_ablation.py)
- [Coveo reduced-budget scoring control](../scripts/qualify_junction_score_coveo.py)
- [Four-workload score validator](../scripts/summarize_junction_scores_four.py)

The eight scripts required for the new Figure 8 and junction-score experiments
were integrated onto main. The code paths use frozen datasets and preserve
the primary RAM-only NavHints design.

## Release gates

1. Check rights for Coveo-derived data, host logs, and third-party sources.
2. Decide on explicit licensing for original project code.
3. Deposit an approved data-minimized evidence package in a durable archive.
4. Verify the public download and a clean offline-reproduction run.
5. Freeze code and paper SHAs and replace the artifact URL if an immutable
   release/DOI is published.

The original 14 frozen Actions ZIPs were independently checksum-validated
on October 17 before their expected October 25-27 expiry. They are preserved
separately for author-only access. A sanitized summary-only candidate exists,
but neither original logs nor raw licensed input vectors have been released
through this repository.

The research-paper artifact URL currently points to this public GeoIVF
repository, without claiming that every archived experiment is permanently
deposited here.
