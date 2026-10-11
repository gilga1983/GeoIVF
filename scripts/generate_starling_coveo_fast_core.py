#!/usr/bin/env python3
"""Build matched classic-vs-fast Core protocol on original Starling and Coveo.

Inherits the previously validated original-author source/index/SSD/Coveo
5K static train, 20K causal warmup, 5K query eval. Only the optional
query-side scoring kernel differs. 3 repetitions, 4 widths, rotated arms.
"""
from pathlib import Path
import os
import subprocess
import sys

def once(s,old,new,label):
    n=s.count(old)
    if n!=1:raise RuntimeError(f"{label}: expected one anchor, saw {n}")
    return s.replace(old,new,1)

def main():
    subprocess.run(["python3","scripts/generate_starling_coveo_coverage.py"],
                   check=True)
    s=Path("scripts/run_starling_coveo_coverage.generated.sh").read_text()
    s=once(s,
       'ART="$PWD/artifacts/starling-coveo-coverage"',
       'ART="$PWD/artifacts/starling-coveo-fast-core"',
       "distinct archived fast experiment")
    s=once(s,
       'WORK="$RUNNER_TEMP/starling-coveo-coverage-$GITHUB_RUN_ID"',
       'WORK="$RUNNER_TEMP/starling-coveo-fast-core-$GITHUB_RUN_ID"',
       "distinct transient index source")
    s=once(s,
       'python3 scripts/patch_starling_coveo_coverage.py "$root"',
       'python3 scripts/patch_starling_coveo_coverage.py "$root"\n'
       '        python3 scripts/patch_starling_fast_core.py "$root"',
       "apply fast identical-semantics native scorer")
    for old,new in [
      ("0) order=(baseline recent512 core16k_recent512 diverse_core maxmin_core cover_core gate50_cover) ;;",
       "0) order=(baseline recent512 fast_recent512 core16k_recent512 fast_core) ;;"),
      ("1) order=(cover_core gate50_cover baseline maxmin_core recent512 diverse_core core16k_recent512) ;;",
       "1) order=(fast_core core16k_recent512 baseline fast_recent512 recent512) ;;"),
      ("2) order=(recent512 maxmin_core core16k_recent512 gate50_cover baseline cover_core diverse_core) ;;",
       "2) order=(core16k_recent512 fast_recent512 fast_core recent512 baseline) ;;"),
    ]:
        s=once(s,old,new,"5-arm balanced native execution sequence")
    s=once(s,'STARLING_COVEO_COVERAGE_SUCCESS',
           'STARLING_COVEO_FAST_CORE_SUCCESS',"unique finish marker")
    s=once(s,
       'python3 scripts/summarize_starling_coveo_coverage.py',
       'python3 scripts/summarize_starling_fast_core.py',
       "strict query-by-query equivalence and paired timing analysis")
    p=Path("scripts/run_starling_coveo_fast_core.generated.sh")
    p.write_text(s)
    p.chmod(0o755)
    subprocess.run(["bash","-n",str(p)],check=True)
    subprocess.run(
        ["python3","scripts/patch_starling_fast_core.py","--self-test"],
        check=True)
    print("STARLING_COVEO_FAST_GENERATED 3reps x 5arms x 4widths "
          "train5000,warm20000,eval5000 pairwise-hard-correctness",flush=True)
    if "--run" in sys.argv:
        os.execvp("bash",["bash",str(p)])

if __name__=="__main__": main()
