#!/usr/bin/env python3
"""Same-session equal-routing-payload NavHints Core vs author-aligned CatapultDB.

Compares *original* NavHints (16K junctions / Recent512 / 1 extra start),
*reallocated* NavHints (14,464 junctions / Recent2048 / 8 starts), and the
author-aligned CatapultDB LSH-bucket routing policy on one pinned DiskANN3 SSD.

All arms: 5K disjoint static training history; 4K causal online warmup;
final 1K measured, strict chronological per-query feedback. Catapult
learns a separate causal prefix snapshot per (L,hash seed). 3 rotated
repetitions, Catapult seeds 0/1/2, matched Recall@10 without extrapolation.
The author's original in-memory graph search is NOT being benchmarked.
"""
from pathlib import Path
import argparse, os, json, hashlib, statistics

from qualify_navhints_catapult_headtohead import (
    run_one, shape, slice_fbin, snapshot_entries,
    TRAIN,WARM,MEASURE,DIM,SEEDS,HASH_BITS,CATAPULT_CAPACITY,
    SEARCH_THREADS,BEAM,K
)

LS=(10,12,20,24,40,80,120,160,200,240,280,320)
NAV_ARMS={
    "core_original": {"junctions":16000,"cache":512,"starts":1},
    "core_reallocated": {"junctions":14464,"cache":2048,"starts":8},
}
PAYLOAD_BYTES=102940

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")

def monotone(curve):
    ordered=sorted(({"L":int(l),**v} for l,v in curve.items()),
                   key=lambda x:(x["recall"],x["reads"]))
    out=[]
    for v in ordered:
        if out and v["recall"] <= out[-1]["recall"]+1e-7:
            if v["reads"]<out[-1]["reads"]:
                out[-1]=v
        else:
            out.append(v)
    return out

def interp(curve,target):
    pts=monotone(curve)
    if not pts or target<pts[0]["recall"]-1e-7 or target>pts[-1]["recall"]+1e-7:
        return None
    for p in pts:
        if abs(target-p["recall"])<1e-7:
            return {"lower_L":p["L"],"upper_L":p["L"],
                    "reads":p["reads"],"latency_us":p["latency_us"]}
    for p,q in zip(pts,pts[1:]):
        if p["recall"]<target<q["recall"]:
            f=(target-p["recall"])/(q["recall"]-p["recall"])
            return {"lower_L":p["L"],"upper_L":q["L"],
                    "reads":p["reads"]+f*(q["reads"]-p["reads"]),
                    "latency_us":p["latency_us"]+f*(q["latency_us"]-p["latency_us"])}
    return None

def summarize(measured,repetitions):
    curves={}
    for rep in range(repetitions):
        for arm in NAV_ARMS:
            curves[rep,arm]={str(l):measured[f"r{rep}-L{l}-{arm}"] for l in LS}
        for seed in SEEDS:
            curves[rep,f"cat-s{seed}"]={
                str(l):measured[f"r{rep}-L{l}-cat-s{seed}"] for l in LS}
    report=[]
    for L in LS:
        target=statistics.mean(curves[rep,"core_original"][str(L)]["recall"]
                               for rep in range(repetitions))
        data={"target_original_L":L,"recall_at_10":target,"available":False}
        records={}
        for arm in NAV_ARMS:
            series=[interp(curves[rep,arm],target) for rep in range(repetitions)]
            if any(x is None for x in series):break
            records[arm]=series
        if len(records)!=len(NAV_ARMS):
            data["reason"]="one NavHints curve does not reach recall target"
            report.append(data);continue
        cat=[]
        for rep in range(repetitions):
            for seed in SEEDS:
                x=interp(curves[rep,f"cat-s{seed}"],target)
                if x is None:break
                cat.append({"rep":rep,"seed":seed,**x})
            if len(cat)!=(rep+1)*len(SEEDS):break
        if len(cat)!=repetitions*len(SEEDS):
            data["reason"]="a Catapult hash seed does not reach recall target"
            report.append(data);continue
        old_reads=statistics.mean(x["reads"] for x in records["core_original"])
        new_reads=statistics.mean(x["reads"] for x in records["core_reallocated"])
        cat_reads=statistics.mean(x["reads"] for x in cat)
        if min(old_reads,new_reads,cat_reads)<=0:raise ValueError("invalid logical SSD counts")
        data.update({
            "available":True,
            "original_reads":old_reads,
            "reallocated_reads":new_reads,
            "catapult_reads":cat_reads,
            "new_saves_vs_original_percent":100*(1-new_reads/old_reads),
            "new_saves_vs_catapult_percent":100*(1-new_reads/cat_reads),
            "original_saves_vs_catapult_percent":100*(1-old_reads/cat_reads),
            "nav_original_interpolants":records["core_original"],
            "nav_reallocated_interpolants":records["core_reallocated"],
            "catapult_repetition_seed_interpolants":cat
        })
        report.append(data)
    q=[x for x in report if x["available"]]
    if len(q)<5:raise RuntimeError("fewer than five common recall anchors")
    return {
        "target_grid":"original Core's measured recalls at the listed L values",
        "matched_points":report,
        "qualified_anchors":len(q),
        "mean_new_saves_vs_original_percent":statistics.mean(
            x["new_saves_vs_original_percent"] for x in q),
        "mean_new_saves_vs_catapult_percent":statistics.mean(
            x["new_saves_vs_catapult_percent"] for x in q),
        "median_new_saves_vs_catapult_percent":statistics.median(
            x["new_saves_vs_catapult_percent"] for x in q),
        "limitations":"same deterministic queries in all 3 timing repetitions; no independent query sampling or inferential uncertainty; per-query feedback timing not harmonized for latency; do not compare QPS counters across implementations"
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("nav-bin","cat-bin","queries","train-gt","heldout-gt",
              "index-prefix","ivf-original","ivf-new","work","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for k in ("nav_bin","cat_bin","queries","train_gt","heldout_gt",
              "index_prefix","ivf_original","ivf_new","work","out"):
        setattr(a,k,getattr(a,k).resolve())
    if a.reps!=3 or shape(a.queries)!=(TRAIN+WARM+MEASURE,DIM):
        raise ValueError("frozen causal 5K/4K/1K split and 3 reps required")
    for p in (a.nav_bin,a.cat_bin,a.train_gt,a.heldout_gt,a.ivf_original,a.ivf_new):
        if not p.is_file():raise FileNotFoundError(p)
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    ivfs={"core_original":a.ivf_original,"core_reallocated":a.ivf_new}
    budget={}
    for arm,cfg in NAV_ARMS.items():
        ivf=ivfs[arm]
        manifest=json.loads(ivf.with_suffix(ivf.suffix+".manifest.json").read_text())
        if int(manifest["nlist"])!=512 or int(manifest["landmark_ids"])!=cfg["junctions"]:
            raise RuntimeError(f"{arm} trained directory shape mismatch")
        val=ivf.stat().st_size+512*(64+4)+cfg["cache"]*4
        if val!=PAYLOAD_BYTES:
            raise RuntimeError(f"{arm} payload not equal to original: {val}")
        budget[arm]={
            "payload_bytes":val,"landmarks":cfg["junctions"],
            "recent_ids_capacity":cfg["cache"],"injected_starts":cfg["starts"],
            "ivf_sha256":hashlib.sha256(ivf.read_bytes()).hexdigest()}
    cat_payload=(1<<HASH_BITS)*CATAPULT_CAPACITY*4+HASH_BITS*DIM*4
    if cat_payload<PAYLOAD_BYTES:raise RuntimeError("Catapult comparator undersized")
    train=slice_fbin(a.queries,a.work/"train5000.fbin",0,TRAIN)
    held=slice_fbin(a.queries,a.work/"heldout5000.fbin",TRAIN,WARM+MEASURE)
    old_cores=sorted(os.sched_getaffinity(0))
    if len(old_cores)<SEARCH_THREADS:raise RuntimeError("require self-hosted 4 cores")
    os.sched_setaffinity(0,set(old_cores[:SEARCH_THREADS]))
    measurements={}
    snapshots={}
    try:
        for L in LS:
            for seed in SEEDS:
                label=f"cat-train-L{L}-seed{seed}"
                snapshot=a.work/(label+".snapshot")
                run_one(a.cat_bin,label,train,a.train_gt,a.index_prefix,L,a.out,
                        mode="catapult",seed=seed,snapshot_dump=snapshot,train=True)
                occ=snapshot_entries(snapshot)
                if not 0<occ <= (1<<HASH_BITS)*CATAPULT_CAPACITY:
                    raise RuntimeError("Invalid bucket count after chronological history")
                snapshots[label]={"entries":occ,"sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest()}
                save(a.out/"training.partial.json",snapshots)
        for rep in range(a.reps):
            grid=list(LS)
            grid=grid[rep:]+grid[:rep]
            for ix,L in enumerate(grid):
                arms=list(NAV_ARMS)+[f"cat-s{seed}" for seed in SEEDS]
                shift=(ix+rep)%len(arms)
                arms=arms[shift:]+arms[:shift]
                print(f"REP={rep} L={L} order={','.join(arms)}",flush=True)
                for arm in arms:
                    name=f"r{rep}-L{L}-{arm}"
                    if arm.startswith("core_"):
                        cfg=NAV_ARMS[arm]
                        data=run_one(
                            a.nav_bin,name,held,a.heldout_gt,a.index_prefix,L,a.out,
                            mode="nav",ivf=ivfs[arm],
                            extra_env={
                                "DISKANN_EXPERIENCE_CACHE_CAPACITY":cfg["cache"],
                                "DISKANN_EXPERIENCE_HUB_CAPACITY":0,
                                "DISKANN_LOCAL_CACHE_MODE":"fifo",
                                "DISKANN_LOCAL_RECENT_STARTS":cfg["starts"],
                            })
                    else:
                        seed=int(arm.rsplit("s",1)[-1])
                        data=run_one(
                            a.cat_bin,name,held,a.heldout_gt,a.index_prefix,L,a.out,
                            mode="catapult",seed=seed,
                            snapshot_load=a.work/f"cat-train-L{L}-seed{seed}.snapshot")
                    measurements[name]=data
                    save(a.out/"results.partial.json",measurements)
    finally:
        os.sched_setaffinity(0,set(old_cores))

    report={
        "title":"original versus reallocated equal-payload NavHints Core vs author-aligned Catapult-on-DiskANN",
        "source":"pinned Microsoft DiskANN SSD graph, Catapult original-author semantics adapted to SSD; NOT native Catapult's in-memory search engine",
        "query_split":"PubMed/MedCPT 5K static history +4K strictly causal online warmup +1K measured; same deterministic requests",
        "metric":"logical SSD reads at matched recall@10, K=10 beam=8, same NVMe",
        "timing_limitations":"different per-query feedback timing, SSD latency noise, no valid aggregate latency claim",
        "widths":list(LS),
        "repetitions":a.reps,"catapult_hash_seeds":list(SEEDS),
        "auxiliary_payload_navhints":budget,
        "auxiliary_payload_catapult":cat_payload,
        "allocation_warning":"Equal ID/map payload, not yet equal process RSS including sets/allocators",
        "catapult_snapshots":snapshots,
        "measurements":measurements,
        "summary":summarize(measurements,a.reps),
    }
    save(a.out/"navhints-optimal-core-vs-catapult.json",report)
    print("NAVHINTS_OPTIMAL_CORE_AUTHOR_CATAPULT_CAEPOCH_PASS",flush=True)
    print(json.dumps(report["summary"],indent=2),flush=True)

if __name__=="__main__":
    main()
