#!/usr/bin/env python3
"""Evaluate DiskANN after RAM-only PQ graph navigation from the same NavHint seed."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess,time
from pathlib import Path
import numpy as np

LS=(10,20,40,80,160,320)
K=10
THREADS=4
BEAM=8

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2)+"\n")

def fbin_shape(p):
    with p.open("rb") as f: r,d=struct.unpack("<II",f.read(8))
    return r,d

def suffix_fbin(src,dst,start):
    r,d=fbin_shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*d*4); fo.write(struct.pack("<II",r-start,d))
        while True:
            b=fi.read(16<<20)
            if not b: break
            fo.write(b)

def suffix_gt(src,dst,start):
    raw=src.read_bytes(); r,k=struct.unpack("<II",raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II",r-start,k)); f.write(raw[8+start*k*4:])

def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o: out.append(o)
        else:
            for v in o.values(): out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(result_rows(v))
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
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    with logp.open("w") as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS): raise ValueError("bad L grid")
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rows=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rows])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rows])),
            "latency_us":float(np.median([float(r["mean_latency"]) for r in rows])),
            "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rows])),
            "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rows])),
        }
    return out

def interp(summary,target,field):
    pts=[]
    best=-1e9
    for l in LS:
        r=summary[str(l)]["recall_percent"]
        if r>=best-1e-9: pts.append((r,summary[str(l)])); best=max(best,r)
    if target<pts[0][0] or target>pts[-1][0]: return None
    for (a,ra),(b,rb) in zip(pts,pts[1:]):
        if a<=target<=b:
            t=0 if b==a else (target-a)/(b-a)
            return ra[field]+t*(rb[field]-ra[field])
    return pts[-1][1][field]

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","seed-dir","manifest","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","seed_dir","manifest","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    held=args.work/"held.fbin"; gt=args.work/"held.gt"; suffix_fbin(args.queries,held,9000); suffix_gt(args.gt5000,gt,4000)
    m=json.loads(args.manifest.read_text())
    configs={"navhint":args.seed_dir/m["baseline_file"]}
    walk_us={"navhint":0.0}
    for key,v in m["configs"].items():
        configs[key]=args.seed_dir/v["file"]; walk_us[key]=float(v["walk_us_per_query"])
    # Keep screen focused.
    wanted=["navhint","b16_s1","b16_s2","b16_s4","b32_s1","b32_s2","b32_s4","b64_s1","b64_s2","b64_s4","b128_s1","b128_s2","b128_s4"]
    methods=[x for x in wanted if x in configs]
    runs={x:[] for x in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(methods); order=methods[shift:]+methods[:shift]
            for method in order:
                rr=run_one(args.binary,args.out,f"r{rep}-{method}",held,gt,args.index_prefix,configs[method]); runs[method].append(rr)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(f"{method} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.3f} lat={float(r40['mean_latency']):.1f}",flush=True)
    finally:
      os.sched_setaffinity(0,set(allowed))
    summary={x:aggregate(runs[x]) for x in methods}; base=summary["navhint"]
    same={}
    for l in LS:
        same[str(l)]={}
        for method in methods:
            r=summary[method][str(l)]; b=base[str(l)]
            same[str(l)][method]={
                "recall_delta_points":r["recall_percent"]-b["recall_percent"],
                "io_change_percent":100*(r["mean_ios"]/b["mean_ios"]-1),
                "diskann_latency_change_percent":100*(r["latency_us"]/b["latency_us"]-1),
                "adjusted_latency_us":r["latency_us"]+walk_us[method],
                "adjusted_latency_change_percent":100*((r["latency_us"]+walk_us[method])/b["latency_us"]-1),
                "cpu_change_percent":100*(r["cpu_us"]/b["cpu_us"]-1),
            }
    fixed={}
    for target in (35.,45.,55.,65.,72.,75.):
        bio=interp(base,target,"mean_ios"); blat=interp(base,target,"latency_us")
        if bio is None or blat is None: continue
        fixed[str(target)]={}
        for method in methods:
            io=interp(summary[method],target,"mean_ios"); lat=interp(summary[method],target,"latency_us")
            if io is None or lat is None: continue
            fixed[str(target)][method]={
                "io_change_percent":100*(io/bio-1),
                "diskann_latency_change_percent":100*(lat/blat-1),
                "adjusted_latency_change_percent":100*((lat+walk_us[method])/blat-1),
            }
    ranking=[]
    for method in methods:
        if method=="navhint": continue
        vals=[fixed[t][method]["adjusted_latency_change_percent"] for t in fixed if method in fixed[t]]
        ios=[fixed[t][method]["io_change_percent"] for t in fixed if method in fixed[t]]
        if vals:
            ranking.append({"method":method,"walk_us_per_query":walk_us[method],
                            "mean_fixed_recall_adjusted_latency_change_percent":float(np.mean(vals)),
                            "mean_fixed_recall_io_change_percent":float(np.mean(ios))})
    ranking.sort(key=lambda x:x["mean_fixed_recall_adjusted_latency_change_percent"])
    result={"policy":"same deployed NavHint bootstrap; optional RAM PQ graph walk; then ordinary DiskANN",
            "ram_walk_manifest":m,"evaluation":{"Ls":list(LS),"reps":args.reps},
            "summary":summary,"same_L":same,"fixed_recall":fixed,"ranking":ranking}
    save(args.out/"ram-pq-walk-runtime.json",result); print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
