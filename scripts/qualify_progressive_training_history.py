#!/usr/bin/env python3
"""Focused training-history refresh for progressive NavHints."""
from __future__ import annotations

import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20)
HISTORIES=(250,500,1000,2500,5000)
BUDGETS=(3000,4096)
RUN_TIMEOUT=600
HEARTBEAT=30

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def shape(path):
    with Path(path).open("rb") as f: raw=f.read(8)
    if len(raw)!=8: raise ValueError("bad fbin header")
    return struct.unpack("<II",raw)

def slice_fbin(src,dst,start,count):
    rows,dim=shape(src)
    if start+count>rows: raise ValueError("slice outside source")
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8+start*dim*4)
        fout.write(struct.pack("<II",count,dim))
        left=count*dim*4
        while left:
            b=fin.read(min(left,16<<20))
            if not b: raise ValueError("truncated")
            fout.write(b); left-=len(b)

def result_rows(obj):
    out=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj: out.append(obj)
        else:
            for v in obj.values(): out.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj: out.extend(result_rows(v))
    return out

def run_one(binary,out,tag,queries,gt,prefix,ivf):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"
    save(inp,cfg)
    env=os.environ.copy()
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE","DISKANN_START_POINTS_FILE","DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE","DISKANN_PAPER_CATAPULT","DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE","DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE","DISKANN_HINT_IVF_NPROBE","DISKANN_HINT_IVF_MAX_STARTS",
        "DISKANN_PROGRESSIVE_HINTS","DISKANN_PROGRESSIVE_HINT_TOPK"):
        env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    env["DISKANN_PROGRESSIVE_HINTS"]="1"
    env["DISKANN_PROGRESSIVE_HINT_TOPK"]="32"
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    started=time.monotonic()
    with logp.open("w") as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        while True:
            try:
                rc=proc.wait(timeout=HEARTBEAT); break
            except subprocess.TimeoutExpired:
                if time.monotonic()-started>=RUN_TIMEOUT:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    raise TimeoutError(tag)
        if rc!=0: raise subprocess.CalledProcessError(rc,cmd)
    rr=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rr]!=list(LS): raise ValueError("bad L grid")
    return rr

def main():
    ap=argparse.ArgumentParser()
    for x in ("binary","queries","gt","index-prefix","state-root","work","out"):
        ap.add_argument("--"+x,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","state_root","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if shape(a.queries)!=(10000,768): raise ValueError("unexpected workload")
    held=a.work/"heldout.fbin"; slice_fbin(a.queries,held,5000,5000)

    states={}
    for meta_path in a.state_root.rglob("state.json"):
        meta=json.loads(meta_path.read_text())
        if meta.get("kind")!="training-size": continue
        h=int(meta["history_rows"]); b=int(meta["requested_budget"])
        if h not in HISTORIES or b not in BUDGETS: continue
        ivf=meta_path.parent/"ivf.bin"
        if ivf.is_file():
            states[(b,h)]=(ivf,meta)
    required=[(3000,h) for h in HISTORIES]+[(4096,h) for h in HISTORIES if h>=500]
    missing=[x for x in required if x not in states]
    if missing: raise ValueError(f"missing fixed-capacity states: {missing}")

    methods=[f"b{b}-h{h}" for b,h in required]
    key_by_name={f"b{b}-h{h}":(b,h) for b,h in required}
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(a.reps):
                shift=rep%len(methods); order=methods[shift:]+methods[:shift]
                for method in order:
                    b,h=key_by_name[method]
                    rr=run_one(a.binary,a.out,f"r{rep}-{method}",held,a.gt,a.index_prefix,states[(b,h)][0])
                    runs[method].append(rr); save(a.out/"runs.partial.json",runs)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={}
    for b in BUDGETS:
        summary[str(b)]={}
        hs=HISTORIES if b==3000 else tuple(h for h in HISTORIES if h>=500)
        for h in hs:
            name=f"b{b}-h{h}"
            summary[str(b)][str(h)]={}
            for l in LS:
                rows=[next(r for r in rr if int(r["search_l"])==l) for rr in runs[name]]
                summary[str(b)][str(h)][str(l)]={
                    "recall_percent":float(np.mean([float(r["recall"]) for r in rows])),
                    "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rows])),
                    "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rows])),
                }

    convergence={}
    for b in BUDGETS:
        ref=summary[str(b)]["5000"]
        convergence[str(b)]={}
        for h,data in summary[str(b)].items():
            convergence[str(b)][h]={}
            for l in map(str,LS):
                convergence[str(b)][h][l]={
                    "recall_delta_vs_h5000_points":data[l]["recall_percent"]-ref[l]["recall_percent"],
                    "io_change_vs_h5000_percent":100*(data[l]["mean_ios"]/ref[l]["mean_ios"]-1),
                    "latency_change_vs_h5000_percent":100*(data[l]["median_latency_us"]/ref[l]["median_latency_us"]-1),
                }

    result={
        "workload":"PubMed1M / MedRAG-Zipf frozen final 5000 queries",
        "policy":"progressive NavHints, K_h=32",
        "fixed_budgets":list(BUDGETS),
        "histories":list(HISTORIES),
        "Ls":list(LS),
        "summary":summary,
        "convergence_vs_5000":convergence,
    }
    save(a.out/"progressive-training-history.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
