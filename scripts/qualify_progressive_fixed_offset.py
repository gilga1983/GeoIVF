#!/usr/bin/env python3
"""Test whether progressive NavHints removes a roughly fixed navigation prefix."""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,12,14,16,18,20,22,24,28,32,36,40,44,48,56,64,72,80,88,96,112,128,144,160,176)
TARGETS=tuple(float(x) for x in range(28,69,2))
HEARTBEAT=30
TIMEOUT=900

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def fbin_shape(path):
    with Path(path).open("rb") as f: raw=f.read(8)
    rows,dim=struct.unpack("<II",raw)
    if Path(path).stat().st_size != 8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def suffix_fbin(src,dst,start):
    rows,dim=fbin_shape(src)
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8+start*dim*4); fout.write(struct.pack("<II",rows-start,dim))
        remaining=(rows-start)*dim*4
        while remaining:
            b=fin.read(min(16<<20,remaining))
            if not b: raise ValueError("truncated fbin")
            fout.write(b); remaining-=len(b)

def result_rows(obj):
    out=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj: out.append(obj)
        else:
            for v in obj.values(): out.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj: out.extend(result_rows(v))
    return out

def run(binary,out,tag,queries,gt,prefix,ivf,progressive):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
            "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
            "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
            "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"
    save(inp,cfg)
    env=os.environ.copy()
    for name in ("DISKANN_SKIP_RECALL","DISKANN_STATIC_CACHE_IDS_FILE","DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE","DISKANN_HINT_IVF_MAX_STARTS","DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_PROGRESSIVE_HINT_TOPK","DISKANN_GLOBAL_START_IDS_FILE","DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE","DISKANN_IP_PORTAL_ROUTER_FILE","DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE"):
        env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if progressive:
        env["DISKANN_PROGRESSIVE_HINTS"]="1"
        env["DISKANN_PROGRESSIVE_HINT_TOPK"]="16"
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    t=time.monotonic()
    with logp.open("w") as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        while True:
            try:
                rc=p.wait(timeout=HEARTBEAT); break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-t
                print(f"heartbeat {tag} {elapsed:.0f}s",flush=True)
                if elapsed>=TIMEOUT:
                    p.terminate()
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise TimeoutError(tag)
        if rc: raise subprocess.CalledProcessError(rc,cmd)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS): raise ValueError("unexpected L rows")
    return [dict(r) for r in rows]

def aggregate(reps):
    out={}
    for l in LS:
        rr=[next(r for r in rows if int(r["search_l"])==l) for rows in reps]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rr])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rr])),
            "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rr])),
        }
    return out

def interp(summary,target,field):
    pts=[(int(l),float(r["recall_percent"]),float(r[field])) for l,r in summary.items()]
    pts.sort()
    pts2=[]; best=-1e99
    for p in pts:
        if p[1]+1e-9>=best:
            pts2.append(p); best=max(best,p[1])
    if target<pts2[0][1] or target>pts2[-1][1]: return None
    for a,b in zip(pts2,pts2[1:]):
        if a[1]<=target<=b[1]:
            if b[1]<=a[1]+1e-12:return b[2]
            t=(target-a[1])/(b[1]-a[1])
            return a[2]+t*(b[2]-a[2])
    return pts2[-1][2] if abs(target-pts2[-1][1])<1e-9 else None

def linfit(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    slope,intercept=np.polyfit(x,y,1)
    pred=intercept+slope*x
    ssr=float(np.sum((y-pred)**2)); sst=float(np.sum((y-y.mean())**2))
    return {"slope":float(slope),"intercept":float(intercept),
            "r2":float(1-ssr/sst) if sst>0 else 0.0}

def main():
    ap=argparse.ArgumentParser()
    for name in ("binary","queries","gt","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","ivf_16k","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if fbin_shape(a.queries)!=(10000,768):raise ValueError("unexpected workload")
    held=a.work/"heldout.fbin"; suffix_fbin(a.queries,held,5000)
    methods=("canonical","progressive16")
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(a.reps):
                order=methods[rep%2:]+methods[:rep%2]
                for m in order:
                    rr=run(a.binary,a.out,f"r{rep}-{m}",held,a.gt,a.index_prefix,a.ivf_16k,m!="canonical")
                    runs[m].append(rr); save(a.out/"runs.partial.json",runs)
    finally:
        os.sched_setaffinity(0,set(allowed))
    summary={m:aggregate(runs[m]) for m in methods}
    matched=[]
    for target in TARGETS:
        vals={}
        for m in methods:
            vals[m]={
                "latency_us":interp(summary[m],target,"median_latency_us"),
                "mean_ios":interp(summary[m],target,"mean_ios"),
            }
        if any(v["latency_us"] is None for v in vals.values()): continue
        matched.append({
            "recall_percent":target,
            "canonical_latency_us":vals["canonical"]["latency_us"],
            "progressive_latency_us":vals["progressive16"]["latency_us"],
            "latency_saved_us":vals["canonical"]["latency_us"]-vals["progressive16"]["latency_us"],
            "latency_saved_percent":100*(1-vals["progressive16"]["latency_us"]/vals["canonical"]["latency_us"]),
            "canonical_ios":vals["canonical"]["mean_ios"],
            "progressive_ios":vals["progressive16"]["mean_ios"],
            "ios_saved":vals["canonical"]["mean_ios"]-vals["progressive16"]["mean_ios"],
            "io_saved_percent":100*(1-vals["progressive16"]["mean_ios"]/vals["canonical"]["mean_ios"]),
        })
    lat_saved=[x["latency_saved_us"] for x in matched]
    io_saved=[x["ios_saved"] for x in matched]
    result={
        "workload":"MedRAG-Zipf heldout 5000",
        "policy":"progressive K_h=16 vs one-shot same 16K directory",
        "Ls":list(LS),"targets":[x["recall_percent"] for x in matched],
        "summary":summary,"matched_recall":matched,
        "absolute_latency_saving":{
            "mean_us":float(np.mean(lat_saved)),"std_us":float(np.std(lat_saved,ddof=1)),
            "min_us":float(np.min(lat_saved)),"max_us":float(np.max(lat_saved)),
            "fit_vs_canonical_latency":linfit([x["canonical_latency_us"] for x in matched],lat_saved),
        },
        "absolute_io_saving":{
            "mean_ios":float(np.mean(io_saved)),"std_ios":float(np.std(io_saved,ddof=1)),
            "min_ios":float(np.min(io_saved)),"max_ios":float(np.max(io_saved)),
            "fit_vs_canonical_ios":linfit([x["canonical_ios"] for x in matched],io_saved),
        }
    }
    save(a.out/"progressive-fixed-offset.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
