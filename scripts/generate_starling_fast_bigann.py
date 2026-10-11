#!/usr/bin/env python3
"""Original Starling BigANN-10M native PQ fast/legacy, identical route comparison.

Reuses successful native BigANN index builder and 5K heldout query protocol:
5K separate training, 4K causal warmup, final 1K measured, 3 repetitions.
No policy changes, no new dataset preprocessing, self-hosted NVMe only.
"""
from pathlib import Path
import os
import subprocess
import sys

def once(s,old,new,why):
    n=s.count(old)
    if n!=1:raise RuntimeError(f"{why}: expected exactly 1, got {n}")
    return s.replace(old,new,1)

def replace_n(s,old,new,n,why):
    if s.count(old)!=n:
        raise RuntimeError(f"{why}: expected {n} anchors, got {s.count(old)}")
    return s.replace(old,new)

def main():
    subprocess.run(["python3","scripts/generate_starling_selective_diverse.py"],check=True)
    s=Path("scripts/run_starling_selective_diverse.generated.sh").read_text()
    s=once(s,
       'ART="$PWD/artifacts/starling-selective-diverse"',
       'ART="$PWD/artifacts/starling-fast-bigann10m"',
       "archived full native BigANN outputs")
    s=once(s,
       'WORK="$RUNNER_TEMP/starling-selective-diverse-$GITHUB_RUN_ID"',
       'WORK="$RUNNER_TEMP/starling-fast-bigann10m-$GITHUB_RUN_ID"',
       "isolated temp source/index")
    s=once(s,
       'python3 scripts/patch_starling_selective_diverse.py "$root"',
       'python3 scripts/patch_starling_selective_diverse.py "$root"\n'
       '        python3 scripts/patch_starling_fast_bigann.py "$root"',
       "apply identical semantics packed PQ native patch")
    for old,new in [
       ("0) order=(baseline core16k_recent512 diverse_core gate50_core gate50_diverse gate25_diverse) ;;",
        "0) order=(baseline recent512 fast_recent512 core16k_recent512 fast_core) ;;"),
       ("1) order=(gate50_diverse gate25_diverse baseline diverse_core gate50_core core16k_recent512) ;;",
        "1) order=(fast_core core16k_recent512 baseline fast_recent512 recent512) ;;"),
       ("2) order=(gate50_core diverse_core core16k_recent512 gate25_diverse baseline gate50_diverse) ;;",
        "2) order=(core16k_recent512 fast_recent512 fast_core recent512 baseline) ;;"),
    ]:
        s=once(s,old,new,"balanced 5-arm source-identical comparisons")
    s=replace_n(s,'LS="20 40 80"','LS="20 40 80 160"',2,
                "extend exact recall frontier to L160")
    s=once(s,
       '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 3',
       '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 4',
       "require full four-width author recall log")
    s=once(s,"STARLING_SELECTIVE_DIVERSE_SUCCESS",
           "STARLING_FAST_BIGANN_SUCCESS","unique complete marker")
    s=once(s,
       "python3 scripts/summarize_starling_selective_diverse.py",
       "python3 scripts/summarize_starling_fast_bigann.py",
       "enforce exact query matching and calculate results")
    p=Path("scripts/run_starling_fast_bigann.generated.sh")
    p.write_text(s);p.chmod(0o755)
    subprocess.run(["bash","-n",str(p)],check=True)
    subprocess.run(["python3","scripts/patch_starling_fast_bigann.py","--self-test"],check=True)
    print("STARLING_FAST_BIGANN_GENERATED n=10M train5K warm4K eval1K "
          "three_rotated_repetitions arms=5 L=20,40,80,160",flush=True)
    if "--run" in sys.argv:
        os.execvp("bash",["bash",str(p)])
if __name__=="__main__":main()
