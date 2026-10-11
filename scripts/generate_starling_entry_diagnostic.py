#!/usr/bin/env python3
"""Generate a tightly scoped native Starling diagnostic run from proven Core CI.

Mutations are exact-anchored and fail closed if the upstream integration
changes. The sole original-author source modification is an instrumentation
layer applied to the already-audited temporary Starling checkout.
"""
from pathlib import Path
import os
import subprocess
import sys


def swap(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main():
    base = Path("scripts/run_starling_navhints_core_ci.sh").read_text()
    base = swap(
        base,
        'ART="$PWD/artifacts/starling-navhints-core"',
        'ART="$PWD/artifacts/starling-entry-diagnostics"',
        "artifact destination",
    )
    base = swap(
        base,
        'WORK="$RUNNER_TEMP/starling-navhints-core-$GITHUB_RUN_ID"',
        'WORK="$RUNNER_TEMP/starling-entry-diagnostics-$GITHUB_RUN_ID"',
        "isolated build scratch",
    )
    base = swap(
        base,
        'python3 scripts/patch_starling_native_junctions.py "$root"',
        'python3 scripts/patch_starling_native_junctions.py "$root"\n'
        '        python3 scripts/patch_starling_entry_diagnostics.py "$root"',
        "apply temporary diagnostic native patch before native build",
    )
    if base.count('LS="20 40 80 160 320 640"') != 2:
        raise RuntimeError("expected Starling and dormant Gorgeous L values")
    base = base.replace('LS="20 40 80 160 320 640"', 'LS="20 40 80"')
    for old, new in (
        ("0) order=(baseline random512 recent512 learned16k core16k_recent512) ;;",
         "0) order=(baseline score_only core16k_recent512 oracle) ;;"),
        ("1) order=(learned16k core16k_recent512 baseline random512 recent512) ;;",
         "1) order=(core16k_recent512 oracle baseline score_only) ;;"),
        ("2) order=(recent512 learned16k baseline core16k_recent512 random512) ;;",
         "2) order=(score_only baseline oracle core16k_recent512) ;;"),
    ):
        base = swap(base,old,new,"rotate four diagnostic arms")
    base = swap(
        base,
        '    (cd "$source_dir/scripts" && STARLING_NAVHINTS_EVAL="$mode" \\',
        '    mkdir -p "$ART/diagnostics/rep$rep-$mode"\n'
        '    (cd "$source_dir/scripts" && STARLING_NAVHINTS_EVAL="$mode" \\\n'
        '     STARLING_NAVHINTS_DIAG_DIR="$ART/diagnostics/rep$rep-$mode" \\',
        "persist per-query diagnostic evidence by run",
    )
    base = swap(
        base,
        '       "$ART/starling-rep$rep-$mode-summary.txt")" -ge 6',
        '       "$ART/starling-rep$rep-$mode-summary.txt")" -ge 3',
        "reduced L frontiers",
    )
    base = swap(
        base,
        'STARLING_NATIVE_NAVHINTS_FULL_CORE_SUCCESS',
        'STARLING_ENTRY_DIAGNOSTIC_SUCCESS',
        "distinct diagnostic completion marker",
    )
    base += '\npython3 scripts/summarize_starling_entry_diagnostics.py "$ART/diagnostics" "$ART/diagnostic-summary.json" "$ART/diagnostic-summary.md"\n'
    p = Path("scripts/run_starling_entry_diagnostics.generated.sh")
    p.write_text(base)
    p.chmod(0o755)
    subprocess.run(["bash", "-n", str(p)], check=True)
    print("DIAGNOSTIC_SH_GENERATED arms=baseline,score_only,core16k_recent512,oracle reps=3 L=20,40,80")
    if len(sys.argv)>1 and sys.argv[1]=="--run":
        os.execvp("bash",["bash",str(p)])


if __name__ == "__main__":
    main()
