#!/usr/bin/env python3
"""Measure the compact RAM PQ walk inside DiskANN, including real Rust CPU time."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)

METHODS=(
    "navhint",
    "d4_b8_s2","d4_b16_s2",
    "d8_b8_s2","d8_b16_s2",
    "d16_b8_s2","d16_b16_s2",
)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2)+"\n")

def fbin_shape(p):
    with p.open("rb") as f: rows,dim=struct.unpack("<II",f.read(8))
    if p.stat().st_size!=8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def suffix_fbin(src,dst,start):
    rows,dim=fbin_shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4); fo.write(struct.pack("<II",rows-start,dim))
        rem=(rows-start)*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)

def suffix_gt(src,dst,start):
    raw=src.read_bytes(); rows,k=struct.unpack("<II",raw[:8])
    if len(raw)!=8+rows*k*4: raise ValueError("bad gt")
    with dst.open("wb") as f:
        f.write(struct.pack("<II",rows-start,k)); f.write(raw[8+start*k*4:])

def result_rows(obj):
    out=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj: out.append(obj)
        else:
            for v in obj.values(): out.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj: out.extend(result_rows(v))
    return out

def parse_method(method,graph_root):
    if method=="navhint": return None,None,None
    # d16_b8_s2
    left,b,seeds=method.split("_")
    degree=int(left[1:]); budget=int(b[1:]); seed_count=int(seeds[1:])
    return graph_root/f"pq-hnsw-level0-d{degree}.bin",budget,seed_count

def run_one(binary,out,tag,queries,gt,index_prefix,ivf,method,graph_root):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(index_prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
            "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
            "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
            "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"; save(inp,cfg)
    env=os.environ.copy()
    for name in list(env):
        if name.startswith("DISKANN_"): env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    graph,budget,seeds=parse_method(method,graph_root)
    if graph is not None:
        env["DISKANN_RAM_PQ_GRAPH"]=str(graph)
        env["DISKANN_RAM_PQ_BUDGET"]=str(budget)
        env["DISKANN_RAM_PQ_SEEDS"]=str(seeds)
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    t=time.monotonic()
    with logp.open("w") as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS): raise ValueError(f"{tag}: bad L grid")
    print(f"completed {tag} elapsed={time.monotonic()-t:.2f}s",flush=True)
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rows=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
            "rounds":len(rows),
            "recall_percent":float(np.mean([float(r["recall"]) for r in rows])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rows])),
            "latency_us":float(np.median([float(r["mean_latency"]) for r in rows])),
            "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rows])),
            "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rows])),
        }
    return out

def monotone(summary):
    pts=[]; best=-1e99
    for l in LS:
        r=summary[str(l)]["recall_percent"]
        if r+1e-9>=best:
            pts.append((r,summary[str(l)])); best=max(best,r)
    return pts

def interp(summary,target,field):
    pts=monotone(summary)
    if target<pts[0][0] or target>pts[-1][0]: return None
    if target==pts[0][0]: return float(pts[0][1][field])
    for (lr,lo),(hr,hi) in zip(pts,pts[1:]):
        if lr<=target<=hr:
            if hr<=lr+1e-12: return float(hi[field])
            a=(target-lr)/(hr-lr)
            return float(lo[field])+a*(float(hi[field])-float(lo[field]))
    return float(pts[-1][1][field])

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","graph-root","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=5)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","graph_root","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)

    held=args.work/"held.fbin"; gt=args.work/"held.gt"
    suffix_fbin(args.queries,held,9000); suffix_gt(args.gt5000,gt,4000)
    graph_bytes={}
    for d in (4,8,16):
        p=args.graph_root/f"pq-hnsw-level0-d{d}.bin"
        if not p.is_file(): raise ValueError(f"missing {p}")
        graph_bytes[str(d)]=p.stat().st_size

    runs={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS: raise RuntimeError("need four CPUs")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift=rep%len(METHODS); order=METHODS[shift:]+METHODS[:shift]
                print(f"rep={rep} order={' '.join(order)}",flush=True)
                for method in order:
                    rr=run_one(args.binary,args.out,f"r{rep}-{method}",held,gt,args.index_prefix,args.ivf_16k,method,args.graph_root)
                    runs[method].append(rr); save(args.out/"runs.partial.json",runs)
                    r20=next(x for x in rr if int(x["search_l"])==20)
                    print(f"{method} L20 recall={float(r20['recall']):.3f} io={float(r20['mean_ios']):.3f} lat={float(r20['mean_latency']):.1f} cpu={float(r20['mean_cpu_time']):.1f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}; base=summary["navhint"]
    same={}
    for l in LS:
        b=base[str(l)]; same[str(l)]={}
        for m in METHODS:
            r=summary[m][str(l)]
            same[str(l)][m]={
                "recall_delta_points":r["recall_percent"]-b["recall_percent"],
                "io_change_percent":100*(r["mean_ios"]/b["mean_ios"]-1),
                "latency_change_percent":100*(r["latency_us"]/b["latency_us"]-1),
                "cpu_change_percent":100*(r["cpu_us"]/b["cpu_us"]-1),
                "comparison_delta":r["comparisons"]-b["comparisons"],
            }

    fixed={}
    targets=(35.,45.,55.,65.,72.,75.)
    for target in targets:
        bio=interp(base,target,"mean_ios"); blat=interp(base,target,"latency_us"); bcpu=interp(base,target,"cpu_us")
        if bio is None or blat is None: continue
        fixed[str(target)]={}
        for m in METHODS:
            io=interp(summary[m],target,"mean_ios"); lat=interp(summary[m],target,"latency_us"); cpu=interp(summary[m],target,"cpu_us")
            if io is None or lat is None: continue
            fixed[str(target)][m]={
                "io_change_percent":100*(io/bio-1),
                "latency_change_percent":100*(lat/blat-1),
                "cpu_change_percent":100*(cpu/bcpu-1),
            }

    score=[]
    early_targets=("35.0","45.0","55.0")
    for m in METHODS:
        if m=="navhint": continue
        degree=int(m.split("_")[0][1:])
        vals=[fixed[t][m] for t in early_targets if t in fixed and m in fixed[t]]
        score.append({
            "method":m,
            "graph_bytes":graph_bytes[str(degree)],
            "graph_mib":graph_bytes[str(degree)]/(1024*1024),
            "mean_early_io_change_percent":float(np.mean([v["io_change_percent"] for v in vals])),
            "mean_early_latency_change_percent":float(np.mean([v["latency_change_percent"] for v in vals])),
            "mean_early_cpu_change_percent":float(np.mean([v["cpu_change_percent"] for v in vals])),
        })
    score.sort(key=lambda x:x["mean_early_latency_change_percent"])

    result={
        "workload":"MedRAG-Zipf final 1000; exact PubMed1M IP truth",
        "policy":"canonical 16K Hint-IVF then optional integrated compact RAM PQ walk then ordinary DiskANN",
        "ram_graph_bytes":graph_bytes,
        "evaluation":{"Ls":list(LS),"beam":BEAM,"threads":THREADS,"repetitions":args.reps},
        "summary":summary,
        "same_L":same,
        "fixed_recall":fixed,
        "benefit_per_memory":score,
    }
    save(args.out/"integrated-ram-pq-walk.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
