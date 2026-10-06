#!/usr/bin/env python3
"""Compare canonical one-shot NavHints with progressive retained runner-ups."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from qualify_staged_navhints import (
    BEAM, K, LS, THREADS, aggregate, fbin_shape, interp, payload, run_one, save, suffix_fbin
)


def main():
    ap=argparse.ArgumentParser()
    for name in ("binary","queries","gt","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for name in ("binary","queries","gt","index_prefix","ivf_16k","work","out"):
        setattr(args,name,getattr(args,name).resolve())
    args.work.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)

    if fbin_shape(args.queries)!=(10000,768):
        raise ValueError("unexpected query workload")
    held=args.work/"heldout.fbin"
    suffix_fbin(args.queries,held,5000)

    methods=("canonical16k","progressive16k")
    runs={m:[] for m in methods}

    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(args.reps):
                order=methods[rep%2:]+methods[:rep%2]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for method in order:
                    rr=run_one(
                        args.binary,args.out,f"r{rep}-{method}",
                        held,args.gt,args.index_prefix,
                        start_ivf=args.ivf_16k,start_probe=8,
                        progressive_hints=(method=="progressive16k"),
                    )
                    runs[method].append(rr)
                    save(args.out/"runs.partial.json",runs)
                    print(
                        f"finished rep={rep} method={method} "
                        f"L10 recall={float(rr[0]['recall']):.3f} "
                        f"ios={float(rr[0]['mean_ios']):.3f} "
                        f"lat={float(rr[0]['mean_latency']):.1f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in methods}
    memory=payload(args.ivf_16k)

    fixed={}
    lo=max(summary[m]["10"]["recall_percent"] for m in methods)
    hi=min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0,45.0,55.0,65.0):
        if not(lo<=target<=hi):
            continue
        item={}
        for m in methods:
            item[m]={
                "mean_ios":interp(summary[m],target,"mean_ios"),
                "latency_us":interp(summary[m],target,"median_latency_us"),
                "runtime_payload_bytes":memory,
            }
        item["progressive_vs_canonical"]={
            "io_change_percent":100*(item["progressive16k"]["mean_ios"]/item["canonical16k"]["mean_ios"]-1),
            "latency_change_percent":100*(item["progressive16k"]["latency_us"]/item["canonical16k"]["latency_us"]-1),
        }
        fixed[str(target)]=item

    same_l={}
    for l in LS:
        a=summary["canonical16k"][str(l)]
        b=summary["progressive16k"][str(l)]
        same_l[str(l)]={
            "canonical_recall":a["recall_percent"],
            "progressive_recall":b["recall_percent"],
            "recall_delta_points":b["recall_percent"]-a["recall_percent"],
            "canonical_ios":a["mean_ios"],
            "progressive_ios":b["mean_ios"],
            "io_change_percent":100*(b["mean_ios"]/a["mean_ios"]-1),
            "canonical_latency_us":a["median_latency_us"],
            "progressive_latency_us":b["median_latency_us"],
            "latency_change_percent":100*(b["median_latency_us"]/a["median_latency_us"]-1),
            "canonical_cpu_us":a["median_cpu_us"],
            "progressive_cpu_us":b["median_cpu_us"],
        }

    result={
        "workload":"MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "vocabulary":"unchanged canonical 16K NavHints directory",
        "progressive_policy":{
            "routing":"same initial Hint-IVF routing pass",
            "retained":"top 32 already-scored hint candidates",
            "start":"best retained candidate only",
            "later":"after each natural DiskANN beam, offer at most one retained unvisited hint through the ordinary fixed-L admission gate",
            "extra_pq_routing":0,
            "extra_persistent_bytes":0,
            "beam_splitting":False,
        },
        "runtime_payload_bytes":memory,
        "evaluation":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS,"repetitions":args.reps},
        "summary":summary,
        "same_L_comparison":same_l,
        "fixed_recall_comparison":fixed,
    }
    save(args.out/"progressive-navhints.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
