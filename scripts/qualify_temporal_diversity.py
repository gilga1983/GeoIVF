#!/usr/bin/env python3
"""Compare canonical progressive NavHints with an equal-memory temporal portfolio."""
from __future__ import annotations
import argparse, fcntl, json, os
from pathlib import Path

from qualify_staged_navhints import (
    BEAM, K, LS, THREADS, aggregate, fbin_shape, interp, payload, run_one, save, suffix_fbin
)

def main():
    ap=argparse.ArgumentParser()
    for name in ("binary","queries","gt","index-prefix","ivf-canonical","ivf-temporal","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","ivf_canonical","ivf_temporal","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if fbin_shape(a.queries)!=(10000,768): raise ValueError("unexpected query workload")
    held=a.work/"heldout.fbin"; suffix_fbin(a.queries,held,5000)

    methods=("canonical","temporal")
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS: raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            print(f"waiting for speed-device lock: {lockp}",flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            print("acquired speed-device lock",flush=True)
            for rep in range(a.reps):
                order=methods[rep%2:]+methods[:rep%2]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for m in order:
                    ivf=a.ivf_canonical if m=="canonical" else a.ivf_temporal
                    rr=run_one(
                        a.binary,a.out,f"r{rep}-{m}",held,a.gt,a.index_prefix,
                        start_ivf=ivf,start_probe=8,
                        progressive_hints=True,progressive_hint_topk=16,
                    )
                    runs[m].append(rr); save(a.out/"runs.partial.json",runs)
                    print(f"finished rep={rep} method={m} L10 recall={float(rr[0]['recall']):.3f} ios={float(rr[0]['mean_ios']):.3f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in methods}
    memory={m:payload(a.ivf_canonical if m=="canonical" else a.ivf_temporal) for m in methods}
    same_l={}
    for l in LS:
        c=summary["canonical"][str(l)]; t=summary["temporal"][str(l)]
        same_l[str(l)]={
            "canonical_recall":c["recall_percent"],"temporal_recall":t["recall_percent"],
            "recall_delta_points":t["recall_percent"]-c["recall_percent"],
            "canonical_ios":c["mean_ios"],"temporal_ios":t["mean_ios"],
            "io_change_percent":100*(t["mean_ios"]/c["mean_ios"]-1),
            "canonical_latency_us":c["median_latency_us"],"temporal_latency_us":t["median_latency_us"],
            "latency_change_percent":100*(t["median_latency_us"]/c["median_latency_us"]-1),
        }

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
        item["temporal_vs_canonical"]={
            "io_change_percent":100*(item["temporal"]["mean_ios"]/item["canonical"]["mean_ios"]-1),
            "latency_change_percent":100*(item["temporal"]["latency_us"]/item["canonical"]["latency_us"]-1),
        }
        fixed[str(target)]=item

    result={
        "workload":"MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "canonical":"16K current general vocabulary, progressive K_h=16",
        "temporal":"8K general + 4K stage-8 + 2K stage-16 + 2K stage-24 disjoint residual specialists, one ordinary 512-cell IVF, progressive K_h=16",
        "memory_payload_bytes":memory,
        "evaluation":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS,"repetitions":a.reps},
        "summary":summary,
        "same_L_comparison":same_l,
        "fixed_recall_comparison":fixed,
    }
    save(a.out/"temporal-diversity.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
