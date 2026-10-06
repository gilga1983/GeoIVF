#!/usr/bin/env python3
"""Upper-bound semantic-cache screen using precomputed query-specific starts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2)+"\n")

def fbin_shape(p):
    with p.open("rb") as f: r,d=struct.unpack("<II",f.read(8))
    return r,d

def suffix_fbin(src,dst,start):
    r,d=fbin_shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*d*4); fo.write(struct.pack("<II",r-start,d))
        rem=(r-start)*d*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)

def suffix_gt(src,dst,start):
    raw=src.read_bytes(); r,k=struct.unpack("<II",raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II",r-start,k)); f.write(raw[8+start*k*4:])

def result_rows(obj):
    out=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj: out.append(obj)
        else:
            for v in obj.values(): out.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj: out.extend(result_rows(v))
    return out

def run_one(binary,out,tag,queries,gt,index_prefix,start_file):
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
    env["DISKANN_START_POINTS_FILE"]=str(start_file)
    with logp.open("w") as log:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(output)],
                       stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS): raise ValueError(f"{tag}: bad L grid")
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
            "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
            "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rs])),
        }
    return out

def monotone(s):
    out=[]; best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best:
            out.append((r,s[str(l)])); best=max(best,r)
    return out

def interp(s,t,field):
    p=monotone(s)
    if t<p[0][0] or t>p[-1][0]: return None
    for (a,ra),(b,rb) in zip(p,p[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[field])+x*(float(rb[field])-float(ra[field]))
    return float(p[-1][1][field])

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","seed-dir","manifest","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=2)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","seed_dir","manifest","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    held=args.work/"held.fbin"; gt=args.work/"held.gt"
    suffix_fbin(args.queries,held,9000); suffix_gt(args.gt5000,gt,4000)

    meta=json.loads(args.manifest.read_text())
    methods={"navhint":args.seed_dir/"navhint-only.bin"}
    memory={"navhint":0}
    for cap,v in meta["capacities"].items():
        # Focus screen on representative capacities.
        if int(cap) not in (128,512,2048,4000): continue
        for nres,arm in v["arms"].items():
            key=f"c{cap}_r{nres}"
            methods[key]=args.seed_dir/arm["file"]
            memory[key]=int(v["total_payload_bytes_top10"])
    names=list(methods)
    runs={m:[] for m in names}

    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift=rep%len(names); order=names[shift:]+names[:shift]
                print(f"rep={rep} order={' '.join(order)}",flush=True)
                for m in order:
                    rr=run_one(args.binary,args.out,f"r{rep}-{m}",held,gt,args.index_prefix,methods[m])
                    runs[m].append(rr); save(args.out/"runs.partial.json",runs)
                    r20=next(x for x in rr if int(x["search_l"])==20)
                    print(f"{m} L20 recall={float(r20['recall']):.3f} io={float(r20['mean_ios']):.2f} lat={float(r20['mean_latency']):.1f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in names}; base=summary["navhint"]
    fixed={}
    for target in (35.,45.,55.,65.,72.,75.):
        bio=interp(base,target,"mean_ios"); blat=interp(base,target,"latency_us")
        if bio is None or blat is None: continue
        fixed[str(target)]={}
        for m in names:
            io=interp(summary[m],target,"mean_ios"); lat=interp(summary[m],target,"latency_us")
            if io is None or lat is None: continue
            fixed[str(target)][m]={
                "io_change_percent":100*(io/bio-1),
                "latency_change_percent":100*(lat/blat-1),
            }

    score=[]
    for m in names:
        if m=="navhint": continue
        vals=[fixed[t][m] for t in ("35.0","45.0","55.0") if t in fixed and m in fixed[t]]
        score.append({
            "method":m,
            "payload_mib":memory[m]/(1024*1024),
            "mean_early_io_change_percent":float(np.mean([x["io_change_percent"] for x in vals])),
            "mean_early_latency_change_percent":float(np.mean([x["latency_change_percent"] for x in vals])),
        })
    score.sort(key=lambda x:x["mean_early_io_change_percent"])
    result={"semantic_cache_manifest":meta,"evaluation":{"Ls":list(LS),"reps":args.reps},
            "summary":summary,"fixed_recall":fixed,"ranking_by_early_io":score}
    save(args.out/"oracle-semantic-cache.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
