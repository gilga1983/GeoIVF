#!/usr/bin/env python3
"""Beam-width robustness for one-shot vs progressive K_h=16 NavHints."""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
K=10
LS=(10,20,40,80,160)
BEAMS=(4,8,16)
TARGETS=(35.0,45.0,55.0,65.0)
HEARTBEAT=30
TIMEOUT=600

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def shape(path):
    with Path(path).open("rb") as f: raw=f.read(8)
    rows,dim=struct.unpack("<II",raw)
    if Path(path).stat().st_size != 8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def suffix(src,dst,start):
    rows,dim=shape(src)
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8+start*dim*4); fout.write(struct.pack("<II",rows-start,dim))
        left=(rows-start)*dim*4
        while left:
            b=fin.read(min(16<<20,left))
            if not b: raise ValueError("truncated fbin")
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

def run(binary,out,tag,queries,gt,prefix,ivf,beam,progressive):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
            "beam_width":beam,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
            "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
            "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"
    save(inp,cfg)
    env=os.environ.copy()
    for name in ("DISKANN_SKIP_RECALL","DISKANN_HINT_IVF_FILE","DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS","DISKANN_PROGRESSIVE_HINTS","DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE","DISKANN_QSEV_FILE","DISKANN_PAPER_CATAPULT",
        "DISKANN_WAYPOINT_CACHE_FILE"):
        env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if progressive: env["DISKANN_PROGRESSIVE_HINTS"]="1"
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    started=time.monotonic()
    with logp.open("w") as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        while True:
            try: rc=p.wait(timeout=HEARTBEAT); break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-started
                print(f"heartbeat {tag} {elapsed:.0f}s",flush=True)
                if elapsed>=TIMEOUT:
                    p.terminate()
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise TimeoutError(tag)
        if rc: raise subprocess.CalledProcessError(rc,cmd)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows] != list(LS): raise ValueError(f"{tag}: bad L rows")
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rr=[next(r for r in rows if int(r["search_l"])==l) for rows in reps]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rr])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rr])),
            "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rr])),
            "median_qps":float(np.median([float(r["qps"]) for r in rr])),
        }
    return out

def interp(summary,target,field):
    pts=[(int(l),float(r["recall_percent"]),float(r[field])) for l,r in summary.items()]
    pts.sort()
    if target<pts[0][1] or target>pts[-1][1]: return None
    for a,b in zip(pts,pts[1:]):
        if a[1]<=target<=b[1]:
            if b[1]<=a[1]+1e-12:return b[2]
            t=(target-a[1])/(b[1]-a[1]); return a[2]+t*(b[2]-a[2])
    return pts[-1][2] if abs(target-pts[-1][1])<1e-9 else None

def main():
    ap=argparse.ArgumentParser()
    for name in ("binary","queries","gt","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","ivf_16k","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if shape(a.queries)!=(10000,768): raise ValueError("unexpected workload")
    held=a.work/"heldout.fbin"; suffix(a.queries,held,5000)
    methods=[(beam,mode) for beam in BEAMS for mode in ("canonical","progressive")]
    runs={f"bw{b}-{m}":[] for b,m in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for rep in range(a.reps):
                order=methods[rep%len(methods):]+methods[:rep%len(methods)]
                for beam,mode in order:
                    key=f"bw{beam}-{mode}"
                    rr=run(a.binary,a.out,f"r{rep}-{key}",held,a.gt,a.index_prefix,a.ivf_16k,beam,mode=="progressive")
                    runs[key].append(rr); save(a.out/"runs.partial.json",runs)
    finally:
        os.sched_setaffinity(0,set(allowed))
    summary={k:aggregate(v) for k,v in runs.items()}
    matched={}
    for beam in BEAMS:
        c=summary[f"bw{beam}-canonical"]; p=summary[f"bw{beam}-progressive"]
        item={}
        for target in TARGETS:
            ci=interp(c,target,"mean_ios"); pi=interp(p,target,"mean_ios")
            cl=interp(c,target,"median_latency_us"); pl=interp(p,target,"median_latency_us")
            if None in (ci,pi,cl,pl):
                item[str(target)]={"available":False}
            else:
                item[str(target)]={
                    "available":True,
                    "canonical_ios":ci,"progressive_ios":pi,
                    "io_saving_percent":100*(1-pi/ci),
                    "canonical_latency_us":cl,"progressive_latency_us":pl,
                    "latency_saving_percent":100*(1-pl/cl),
                }
        matched[str(beam)]=item
    result={
      "workload":"MedRAG-Zipf heldout 5000","vocabulary":"16K / 512 cells / nprobe=8",
      "policy":"progressive K_h=16","beam_widths":list(BEAMS),"Ls":list(LS),
      "repetitions":a.reps,"summary":summary,"matched_recall":matched}
    save(a.out/"beam-sensitivity.json",result); print(json.dumps(result,indent=2))

if __name__=="__main__":main()
