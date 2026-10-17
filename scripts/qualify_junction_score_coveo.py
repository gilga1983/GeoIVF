#!/usr/bin/env python3
"""Junction-score sensitivity on licensed chronological Coveo demand.

The selected full directory retains *all* 12,169 eligible vertices, so its
score cannot change membership. Evaluate a separate controlled B=8192
ablation to make top-B ranking meaningful. C=2048, k=8 and all other
mechanisms are unchanged. Do NOT compare the 8192-arm result to the
published selected-B=12169 absolute performance or call this an ablation
of the full selected configuration.

Three order-rotated real-SSD trials, 20K causal history warm-up and 5K
measured requests, the original native index and recall-interpolation
procedure (no extrapolation).
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import struct
from pathlib import Path

import numpy as np

from qualify_final_core_coveo import (
    THREADS, BEAM, LS, CFG, aggregate, interp, run, save, xshape
)

MODES=("sum","count","mean")
B=8192
C=2048
K=8

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("binary","dataset-manifest","replay-gt","index-prefix","ivf-root","work","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--reps",type=int,default=3)
    a=p.parse_args()
    if a.reps!=3: raise ValueError("Three rotated repeats required")
    for key in ("binary","dataset_manifest","replay_gt","index_prefix","ivf_root","work","out"):
        setattr(a,key,getattr(a,key).resolve())
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    doc=json.loads(a.dataset_manifest.read_text())
    replay=Path(doc["files"]["replay"])
    online_warmup=int(doc["split"]["online_warmup_rows"])
    measured=int(doc["split"]["measured_rows"])
    if online_warmup!=20000 or measured!=5000 or xshape(replay)[0]!=25000:
        raise ValueError("Expected frozen Coveo 20K+5K chronological replay")
    with a.replay_gt.open("rb") as file:
        gt_rows,k=struct.unpack("<II",file.read(8))
    if gt_rows!=25000 or k<10:
        raise ValueError("Coveo ground truth shape mismatch")
    states={}
    bytesizes=set()
    for mode in MODES:
        path=a.ivf_root/f"ivf-{mode}.bin"
        manifest=path.with_suffix(path.suffix+".manifest.json")
        if not path.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"{mode}: absent rebuilt routing file")
        m=json.loads(manifest.read_text())
        if m.get("nlist")!=512 or m.get("landmark_ids")!=B or m.get("state_bytes")!=path.stat().st_size:
            raise ValueError(f"{mode}: B8192 directory geometry or file size differs")
        states[mode]=path
        bytesizes.add(path.stat().st_size)
    if len(bytesizes)!=1: raise ValueError("Unequal routing-memory footprints")
    variants={mode:[] for mode in MODES}
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS:
        raise ValueError("Not enough CPUs for native Coveo four-thread search")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lock=Path.home()/".cache/geoivf/speed-device.lock"
    lock.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lock.open("w") as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            for rep in range(a.reps):
                order=MODES[rep:]+MODES[:rep]
                for mode in order:
                    rr,stats=run(a.binary,a.out,f"r{rep}-score-{mode}",
                       replay,a.replay_gt,a.index_prefix,
                       states[mode],CFG["core"],online_warmup)
                    if stats is None or set(stats)!=set(map(str,LS)):
                        raise ValueError(f"Missing native stats for {mode}, repetition {rep}")
                    for L in LS:
                        z=stats[str(L)]
                        if z.get("cache_capacity")!=C or z.get("hub_capacity")!=0 \
                          or z.get("writes")!=0 or z.get("eval_writes")!=0:
                            raise ValueError(f"{mode}, L{L}: unexpected persistence/memory")
                    variants[mode].append(rr)
                    selected=next(x for x in rr if int(x["search_l"])==80)
                    print(f"Coveo rep={rep} mode={mode} L80 recall={selected['recall']} reads={selected['mean_ios']}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))
    summary={mode:aggregate(variants[mode]) for mode in MODES}
    matched=[]
    for L in LS:
        target=summary["sum"][str(L)]["recall_percent"]
        costs={mode:interp(summary[mode],target,"mean_ios") for mode in MODES}
        if not all(val is not None and val>0 for val in costs.values()):
            continue
        matched.append({
            "sum_L":L,"recall_percent":target,"reads":costs,
            "sum_vs_count_saving_percent":100*(1-costs["sum"]/costs["count"]),
            "sum_vs_mean_saving_percent":100*(1-costs["sum"]/costs["mean"]),
        })
    if len(matched)<3:
        raise ValueError(f"Insufficient common recall targets: {len(matched)}")
    savings={}
    for mode in ("count","mean"):
        arr=[r[f"sum_vs_{mode}_saving_percent"] for r in matched]
        savings[mode]={"targets":len(matched),"mean_read_saving_pct":float(np.mean(arr)),
                       "min_read_saving_pct":min(arr),"max_read_saving_pct":max(arr)}
    result={
        "dataset":doc["dataset"],"dataset_role":"Coveo B8192 rank-sensitive control; full B12169 retains all candidates",
        "reference":"sum first expansion positions",
        "controller":{"B":B,"nominal_B":14464,"full_selected_B":12169,
                      "C":C,"k":K,"regions":512,"persistent_exits":0},
        "protocol":{"repetitions":a.reps,"train":5000,"online_warmup":online_warmup,
                    "measured":measured,"original_native_graph":True,
                    "comparison":"matched-recall interpolation; no extrapolation",
                    "statistical_scope":"descriptive repeated replays of one chronological demand stream"},
        "search":{"L":list(LS),"beam":BEAM,"threads":THREADS},
        "summary":summary,"matched":matched,"savings_vs_alternative":savings
    }
    save(a.out/"junction-score-ablation.json",result)
    print("COVEO_SCORE_ABLATION_B8192_PASS",json.dumps(savings,indent=2))

if __name__=="__main__":
    main()
