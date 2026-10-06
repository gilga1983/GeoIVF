#!/usr/bin/env python3
"""Mechanism study for progressive NavHints.

Sweeps the retained shortlist size while recording, per search L:
  * number of runner-up hints admitted per query,
  * exact natural-beam boundaries where hints were considered/admitted,
  * conditional admission rate at each boundary.

K_h includes the start winner, so K_h=1 is the progressive code-path control
with no usable runner-up.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from qualify_staged_navhints import (
    BEAM, K, LS, THREADS, aggregate, fbin_shape, interp, payload, run_one, save, suffix_fbin
)

TOPKS=(1,2,4,8,16,32)


def mechanism_row(row, num_queries=5000):
    admissions=list(row.get("progressive_hint_hop_histogram", []))
    opportunities=list(row.get("progressive_hint_opportunity_hop_histogram", []))
    n=max(len(admissions),len(opportunities))
    admissions += [0]*(n-len(admissions))
    opportunities += [0]*(n-len(opportunities))
    boundaries=[]
    for hop,(a,o) in enumerate(zip(admissions,opportunities)):
        if o:
            boundaries.append({
                "hops":hop,
                "beam_index":None if hop == 0 else (hop + BEAM - 1)//BEAM,
                "admissions":int(a),
                "opportunities":int(o),
                "conditional_admission_percent":100.0*a/o,
                "admissions_per_query_percent":100.0*a/num_queries,
            })
    counts=[int(x) for x in row.get("progressive_hint_count_histogram",[])]
    return {
        "mean_admissions_per_query":float(row.get("progressive_hint_mean_admissions",0.0)),
        "queries_with_admission_percent":float(row.get("progressive_hint_queries_with_admission_percent",0.0)),
        "admission_count_histogram":counts,
        "admission_count_percent":[100.0*x/num_queries for x in counts],
        "boundaries":boundaries,
    }


def assert_mechanism_stable(reps, method):
    # Deterministic graph traversal should produce identical mechanism histograms
    # across timing repetitions.
    keys=(
        "progressive_hint_count_histogram",
        "progressive_hint_hop_histogram",
        "progressive_hint_opportunity_hop_histogram",
    )
    for l in LS:
        rows=[next(r for r in rr if int(r["search_l"])==l) for rr in reps]
        for key in keys:
            vals=[r.get(key,[]) for r in rows]
            if any(v != vals[0] for v in vals[1:]):
                raise ValueError(f"{method} L={l}: unstable {key}")


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

    methods=("canonical",)+tuple(f"k{k}" for k in TOPKS)
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS:
        raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            print(f"waiting for speed-device lock: {lockp}",flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            print("acquired speed-device lock",flush=True)
            for rep in range(args.reps):
                shift=rep%len(methods)
                order=methods[shift:]+methods[:shift]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for method in order:
                    progressive=method!="canonical"
                    topk=32 if not progressive else int(method[1:])
                    rr=run_one(
                        args.binary,args.out,f"r{rep}-{method}",
                        held,args.gt,args.index_prefix,
                        start_ivf=args.ivf_16k,start_probe=8,
                        progressive_hints=progressive,
                        progressive_hint_topk=topk,
                    )
                    runs[method].append(rr)
                    save(args.out/"runs.partial.json",runs)
                    r10=rr[0]
                    print(
                        f"finished rep={rep} method={method} "
                        f"L10 recall={float(r10['recall']):.3f} "
                        f"ios={float(r10['mean_ios']):.3f} "
                        f"admit={float(r10.get('progressive_hint_mean_admissions',0)):.3f}",
                        flush=True,
                    )
    finally:
        os.sched_setaffinity(0,set(allowed))

    for method in methods:
        if method!="canonical":
            assert_mechanism_stable(runs[method],method)

    summary={m:aggregate(runs[m]) for m in methods}
    mechanism={}
    for method in methods:
        mechanism[method]={}
        first=runs[method][0]
        for row in first:
            mechanism[method][str(int(row["search_l"]))]=mechanism_row(row)

    # Same-L shortlist knee relative to K_h=32 and canonical.
    same_l={}
    for l in LS:
        item={}
        base=summary["canonical"][str(l)]
        full=summary["k32"][str(l)]
        for method in methods:
            row=summary[method][str(l)]
            item[method]={
                "recall_percent":row["recall_percent"],
                "mean_ios":row["mean_ios"],
                "median_latency_us":row["median_latency_us"],
                "mean_admissions_per_query":mechanism[method][str(l)]["mean_admissions_per_query"],
                "queries_with_admission_percent":mechanism[method][str(l)]["queries_with_admission_percent"],
                "io_change_vs_canonical_percent":100.0*(row["mean_ios"]/base["mean_ios"]-1.0),
                "latency_change_vs_canonical_percent":100.0*(row["median_latency_us"]/base["median_latency_us"]-1.0),
                "io_change_vs_k32_percent":100.0*(row["mean_ios"]/full["mean_ios"]-1.0),
                "recall_delta_vs_k32_points":row["recall_percent"]-full["recall_percent"],
            }
        same_l[str(l)]=item

    fixed={}
    lo=max(summary[m]["10"]["recall_percent"] for m in methods)
    hi=min(summary[m]["160"]["recall_percent"] for m in methods)
    for target in (35.0,45.0,55.0,65.0):
        if not(lo<=target<=hi):
            continue
        item={}
        for method in methods:
            item[method]={
                "mean_ios":interp(summary[method],target,"mean_ios"),
                "latency_us":interp(summary[method],target,"median_latency_us"),
            }
        fixed[str(target)]=item

    result={
        "workload":"MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "vocabulary":"unchanged canonical 16K NavHints directory",
        "shortlist_sizes":list(TOPKS),
        "shortlist_definition":"K_h includes the start winner; K_h=1 has zero runner-ups",
        "mechanism":"route once, start from best hint, then offer at most one retained runner-up after each complete native beam",
        "runtime_payload_bytes":payload(args.ivf_16k),
        "evaluation":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS,"repetitions":args.reps},
        "summary":summary,
        "mechanism_by_L":mechanism,
        "same_L_shortlist_sweep":same_l,
        "fixed_recall_shortlist_sweep":fixed,
    }
    save(args.out/"progressive-mechanism.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
