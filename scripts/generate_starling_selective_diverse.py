#!/usr/bin/env python3
"""Build focused, rotated native Starling gating/diversity experiment.

Reuses the successful, original-author BigANN-10M compilation and replay
path. All modifications are on an experiment branch and a temporary pinned
Starling checkout; no manuscript or production code changes.
"""
import os
from pathlib import Path
import subprocess
import sys


def once(s,old,new,reason):
    count=s.count(old)
    if count!=1:
        raise RuntimeError(f"{reason}: expected one exact anchor, got {count}")
    return s.replace(old,new,1)


def main():
    subprocess.run(["python3","scripts/generate_starling_entry_diagnostic.py"],
                   check=True)
    s=Path("scripts/run_starling_entry_diagnostics.generated.sh").read_text()
    s=once(s,'ART="$PWD/artifacts/starling-entry-diagnostics"',
           'ART="$PWD/artifacts/starling-selective-diverse"',
           "distinct results")
    s=once(s,'WORK="$RUNNER_TEMP/starling-entry-diagnostics-$GITHUB_RUN_ID"',
           'WORK="$RUNNER_TEMP/starling-selective-diverse-$GITHUB_RUN_ID"',
           "isolated temporary files")
    s=once(s,
       'python3 scripts/patch_starling_entry_diagnostics.py "$root"',
       'python3 scripts/patch_starling_entry_diagnostics.py "$root"\n'
       '        python3 scripts/patch_starling_selective_diverse.py "$root"',
       "apply 4-layer code patch before native compilation")
    for old,new in [
       ("0) order=(baseline score_only core16k_recent512 oracle) ;;",
        "0) order=(baseline core16k_recent512 diverse_core gate50_core gate50_diverse gate25_diverse) ;;"),
       ("1) order=(core16k_recent512 oracle baseline score_only) ;;",
        "1) order=(gate50_diverse gate25_diverse baseline diverse_core gate50_core core16k_recent512) ;;"),
       ("2) order=(score_only baseline oracle core16k_recent512) ;;",
        "2) order=(gate50_core diverse_core core16k_recent512 gate25_diverse baseline gate50_diverse) ;;"),
    ]:
        s=once(s,old,new,"balanced rotated 6-arm comparison")
    s=once(s,
       'if [ "$mode" = learned16k ] || [ "$mode" = core16k_recent512 ]; then',
       'if [ "$mode" != baseline ]; then',
       "validate loaded native directory in all hint-enabled modes")
    s=once(s,'STARLING_ENTRY_DIAGNOSTIC_SUCCESS',
           'STARLING_SELECTIVE_DIVERSE_SUCCESS',"distinct success marker")
    s=once(s,
       'python3 scripts/summarize_starling_entry_diagnostics.py',
       'python3 scripts/summarize_starling_selective_diverse.py',
       "paired validation and summary")
    s=once(s,
       'echo "STARLING_FULL_CORE rep=$rep mode=$mode begin=$(date -Is)"',
       'echo "STARLING_SELECTIVE_DIVERSE rep=$rep mode=$mode begin=$(date -Is)"',
       "label ablation logs")
    p=Path("scripts/run_starling_selective_diverse.generated.sh")
    p.write_text(s)
    p.chmod(0o755)
    subprocess.run(["bash","-n",str(p)],check=True)
    print("STARLING_SELECTIVE_DIVERSE_SCRIPT_GENERATED reps=3 "
          "modes=baseline,core16k_recent512,diverse_core,gate50_core,"
          "gate50_diverse,gate25_diverse L=20,40,80",flush=True)
    if "--run" in sys.argv:
        os.execvp("bash",["bash",str(p)])


if __name__=="__main__":main()
