#!/usr/bin/env python3
"""Sweep shared residual-bank checkpoint schedules."""
from __future__ import annotations
import argparse, fcntl, json, os
from pathlib import Path
from qualify_staged_navhints import (
    BEAM,K,LS,THREADS,aggregate,fbin_shape,interp,payload,run_one,save,suffix_fbin
)

SCHEDULES={
    "hop3":"3",
    "hop3-5":"3,5",
    "hop3-5-7":"3,5,7",
    "hop3-5-7-8":"3,5,7,8",
}

def main():
    ap=argparse.ArgumentParser()
    for name in ("binary","queries","gt","index-prefix","ivf-16k","ivf-start8k","ivf-residual4k","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for name in ("binary","queries","gt","index_prefix","ivf_16k","ivf_start8k","ivf_residual4k","work","out"):
        setattr(args,name,getattr(args,name).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    if fbin_shape(args.queries)!=(10000,768): raise ValueError("unexpected workload")
    held=args.work/"heldout.fbin"; suffix_fbin(args.queries,held,5000)

    methods=("start16k",)+tuple(SCHEDULES)
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift=rep%len(methods); order=methods[shift:]+methods[:shift]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for method in order:
                    if method=="start16k":
                        kw=dict(start_ivf=args.ivf_16k,start_probe=8)
                    else:
                        kw=dict(start_ivf=args.ivf_start8k,start_probe=4,
                                stage_ivf=args.ivf_residual4k,stage_probe=4,
                                stage_hops=SCHEDULES[method])
                    rr=run_one(args.binary,args.out,f"r{rep}-{method}",held,args.gt,args.index_prefix,**kw)
                    runs[method].append(rr); save(args.out/"runs.partial.json",runs)
                    print(f"finished rep={rep} method={method} L10 "
                          f"recall={float(rr[0]['recall']):.3f} ios={float(rr[0]['mean_ios']):.3f}",
                          flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in methods}
    mem_start=payload(args.ivf_16k)
    mem_shared=payload(args.ivf_start8k)+payload(args.ivf_residual4k)
    memory={"start16k":mem_start,**{m:mem_shared for m in SCHEDULES}}
    fixed={}
    lo=max(summary[m]["10"]["recall_percent"] for m in methods)
    hi=min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0,45.0,55.0,65.0):
        if not(lo<=target<=hi): continue
        item={}
        for m in methods:
            item[m]={
                "mean_ios":interp(summary[m],target,"mean_ios"),
                "latency_us":interp(summary[m],target,"median_latency_us"),
                "runtime_payload_bytes":memory[m],
            }
        prev="hop3"
        for nxt in ("hop3-5","hop3-5-7","hop3-5-7-8"):
            item[f"{nxt}_vs_{prev}"]={
                "io_change_percent":100*(item[nxt]["mean_ios"]/item[prev]["mean_ios"]-1),
                "latency_change_percent":100*(item[nxt]["latency_us"]/item[prev]["latency_us"]-1),
            }
            prev=nxt
        fixed[str(target)]=item

    result={
        "workload":"MedRAG-Zipf heldout 5000",
        "residual_bank":"4096 disjoint stage-3 IDs, shared and PQ-ranked once/query",
        "schedules":SCHEDULES,
        "memory_payload_bytes":memory,
        "summary":summary,
        "fixed_recall_comparison":fixed,
    }
    save(args.out/"shared-schedule-sweep.json",result)
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
