#!/usr/bin/env python3
"""Current progressive NavHints vs paper-faithful CatapultDB on heldout MedRAG-Zipf.

Catapult gets the same first-5K training history, then loads that snapshot for
heldout evaluation and remains online/adaptive. NavHints is frozen. This setup
therefore favors Catapult while isolating the conceptual difference between
better starts and traversal-wide query-conditioned shortcuts.
"""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,12,20,22,24,40,44,80,88,160,176)
SEEDS=(0,1,2)
HEARTBEAT=30
TIMEOUT=900
DIM=768

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def shape(path):
    with Path(path).open("rb") as f: raw=f.read(8)
    rows,dim=struct.unpack("<II",raw)
    if Path(path).stat().st_size != 8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def slice_fbin(src,dst,start,count):
    rows,dim=shape(src)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad slice")
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8+start*dim*4); fout.write(struct.pack("<II",count,dim))
        left=count*dim*4
        while left:
            b=fin.read(min(16<<20,left))
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

def config(out,queries,gt,prefix,ls):
    return {"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(ls),
            "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
            "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
            "search_io_limit":None,"post_processor":None}}}]}

def run(binary,out,tag,queries,gt,prefix,ls,env_extra,skip_recall=False):
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"
    save(inp,config(out,queries,gt,prefix,ls))
    env=os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL","DISKANN_PAPER_CATAPULT","DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY","DISKANN_CATAPULT_SEED","DISKANN_CATAPULT_SNAPSHOT_LOAD",
        "DISKANN_CATAPULT_SNAPSHOT_DUMP","DISKANN_CATAPULT_FREEZE","DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE","DISKANN_HINT_IVF_MAX_STARTS","DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_GLOBAL_START_IDS_FILE","DISKANN_START_POINTS_FILE","DISKANN_QSEV_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE","DISKANN_WAYPOINT_CACHE_FILE"):
        env.pop(name,None)
    if skip_recall: env["DISKANN_SKIP_RECALL"]="1"
    env.update({k:str(v) for k,v in env_extra.items()})
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
    rr=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rr] != list(ls): raise ValueError(f"{tag}: bad L rows")
    return rr

def snapshot_entries(path):
    raw=Path(path).read_bytes()
    if len(raw)<32 or raw[:8]!=b"GICAT001": raise ValueError("bad snapshot")
    return struct.unpack("<I",raw[28:32])[0]

def aggregate(reps):
    out={}
    for l in LS:
        rr=[next(r for r in rows if int(r["search_l"])==l) for rows in reps]
        out[str(l)]={
            "rounds":len(rr),
            "recall_percent":float(np.mean([float(r["recall"]) for r in rr])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rr])),
            "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rr])),
            "median_qps":float(np.median([float(r["qps"]) for r in rr])),
            "mean_catapult_usage_percent":float(np.mean([float(r.get("catapult_usage_percentage",0)) for r in rr])),
            "mean_catapult_starts":float(np.mean([float(r.get("mean_catapult_starts",0)) for r in rr])),
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
    for name in ("catapult-binary","nav-binary","queries","heldout-gt","training-gt",
                 "index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args()
    for n in ("catapult_binary","nav_binary","queries","heldout_gt","training_gt",
              "index_prefix","ivf_16k","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if shape(a.queries)!=(10000,DIM): raise ValueError("unexpected workload")
    train=a.work/"train.fbin"; held=a.work/"heldout.fbin"
    slice_fbin(a.queries,train,0,5000); slice_fbin(a.queries,held,5000,5000)

    snapshots={}
    snapshot_meta=[]
    for seed in SEEDS:
        snap=a.work/f"catapult-seed{seed}.snapshot"
        run(a.catapult_binary,a.out,f"train-catapult-s{seed}",train,a.training_gt,a.index_prefix,(10,),{
            "DISKANN_PAPER_CATAPULT":"1","DISKANN_CATAPULT_HASHES":"8",
            "DISKANN_CATAPULT_CAPACITY":"40","DISKANN_CATAPULT_SEED":seed,
            "DISKANN_CATAPULT_SNAPSHOT_DUMP":snap,
        },skip_recall=True)
        entries=snapshot_entries(snap)
        snapshots[seed]=snap
        snapshot_meta.append({
            "seed":seed,"snapshot_bytes":snap.stat().st_size,"resident_entries":entries,
            "hyperplane_payload_bytes":8*DIM*4,
            "snapshot_plus_hyperplanes_bytes":snap.stat().st_size+8*DIM*4,
        })

    nav_runs=[]
    cat_runs=[]
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for i,seed in enumerate(SEEDS):
                if i%2==0:
                    order=("cat","nav")
                else:
                    order=("nav","cat")
                for method in order:
                    if method=="nav":
                        rr=run(a.nav_binary,a.out,f"eval-nav-r{i}",held,a.heldout_gt,a.index_prefix,LS,{
                            "DISKANN_HINT_IVF_FILE":a.ivf_16k,
                            "DISKANN_HINT_IVF_NPROBE":"8",
                            "DISKANN_HINT_IVF_MAX_STARTS":"1",
                            "DISKANN_PROGRESSIVE_HINTS":"1",
                        })
                        nav_runs.append(rr)
                    else:
                        rr=run(a.catapult_binary,a.out,f"eval-catapult-s{seed}",held,a.heldout_gt,a.index_prefix,LS,{
                            "DISKANN_PAPER_CATAPULT":"1","DISKANN_CATAPULT_HASHES":"8",
                            "DISKANN_CATAPULT_CAPACITY":"40","DISKANN_CATAPULT_SEED":seed,
                            "DISKANN_CATAPULT_SNAPSHOT_LOAD":snapshots[seed],
                            # updates intentionally remain enabled on heldout
                        })
                        cat_runs.append(rr)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={"navhints":aggregate(nav_runs),"catapult":aggregate(cat_runs)}
    matched={}
    for l in (10,20,40,80,160):
        nav=summary["navhints"][str(l)]; target=nav["recall_percent"]
        ci=interp(summary["catapult"],target,"mean_ios")
        cl=interp(summary["catapult"],target,"median_latency_us")
        matched[str(l)]={
            "recall_percent":target,
            "navhints_mean_ios":nav["mean_ios"],
            "navhints_latency_us":nav["median_latency_us"],
            "catapult_available":ci is not None and cl is not None,
            "catapult_interpolated_mean_ios":ci,
            "catapult_interpolated_latency_us":cl,
            "navhints_io_saving_percent":None if ci is None else 100*(1-nav["mean_ios"]/ci),
            "navhints_latency_saving_percent":None if cl is None else 100*(1-nav["median_latency_us"]/cl),
        }

    result={
        "workload":"MedRAG-Zipf first 5K training / final 5K exact heldout",
        "navhints":{"state":"16K / 512 / nprobe=8 / K_h=16","updates_on_heldout":False,
                    "runtime_payload_bytes":100892},
        "catapult":{"hashes":8,"bucket_capacity":40,"training_L":10,
                    "updates_on_heldout":True,"seeds":list(SEEDS),
                    "snapshots":snapshot_meta,
                    "accounting_note":"snapshot + hyperplane payload; bucket/lock/container overhead excluded, favoring Catapult"},
        "search":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS},
        "summary":summary,"matched_recall_against_navhints":matched,
    }
    save(a.out/"catapult-vs-progressive-navhints.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
