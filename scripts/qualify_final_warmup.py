#!/usr/bin/env python3
"""Warm-up curve for frozen NavHints online state."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4; BEAM=8; K=10
LS=(10,20,40,80,160,320)
HISTORY=(0,250,500,1000,2000,3000,4000)
WINDOW=500
TARGETS=(35.,45.,55.,65.,72.,75.)
STAT_KEYS=("cache_capacity","hub_capacity","sample_denominator","fill","cache_inserts","cache_skips",
"cache_evictions","active_direct_hubs","persisted_hubs","direct_learned","direct_duplicates",
"direct_full","sample_trials","sample_accepts","sample_rejects","fifo_evictions","writes","eval_writes",
"write_slots","write_direct_slots","write_filler_slots","eval_write_slots","eval_write_filler_slots","final_page_slots")

def save(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2)+"\n")
def shape(p):
    with Path(p).open("rb") as f: return struct.unpack("<II",f.read(8))
def slice_fbin(src,dst,start,count):
    src,dst=Path(src),Path(dst); rows,dim=shape(src)
    if start+count>rows: raise ValueError("bad fbin slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4); fo.write(struct.pack("<II",count,dim)); rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)
def slice_gt(src,dst,start,count):
    src,dst=Path(src),Path(dst)
    with src.open("rb") as f: rows,k=struct.unpack("<II",f.read(8))
    rb=k*4; ids=rows*rb; size=src.stat().st_size
    ids_only=size==8+ids; both=size==8+2*ids
    if start+count>rows or not(ids_only or both): raise ValueError("bad gt")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k)); fi.seek(8+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem)); fo.write(b); rem-=len(b)
        if both:
            fi.seek(8+ids+start*rb); rem=count*rb
            while rem:
                b=fi.read(min(8<<20,rem)); fo.write(b); rem-=len(b)
def rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o: out.append(o)
        else:
            for v in o.values(): out.extend(rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(rows(v))
    return out
def parse_stats(p):
    out={}
    for line in Path(p).read_text().splitlines():
        if "EXPERIENCE_STATS " not in line: continue
        kv={}
        for t in line.split("EXPERIENCE_STATS ",1)[1].split():
            if "=" in t:
                k,v=t.split("=",1); kv[k]=int(v)
        if "L" in kv: out[str(kv["L"])]={k:kv[k] for k in STAT_KEYS if k in kv}
    return out
def run(binary,out,tag,q,gt,prefix,ivf,experience,hub,warmup):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),"beam_width":BEAM,
      "recall_at":K,"num_threads":THREADS,"is_flat_search":False,"distance":"inner_product",
      "vector_filters_file":None,"num_nodes_to_cache":None,"search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; op=out/f"{tag}.output.json"; log=out/f"{tag}.log"; save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"): env.pop(n,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf); env["DISKANN_HINT_IVF_NPROBE"]="8"; env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if experience:
        env["DISKANN_EXPERIENCE_REPLAY"]="1"; env["DISKANN_EXPERIENCE_WARMUP"]=str(warmup)
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]="512"; env["DISKANN_EXPERIENCE_HUB_CAPACITY"]=str(hub)
        env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"]="2"; env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="1"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    return rr,(parse_stats(log) if experience else {})
def mono(rr):
    pts=[]; best=-1e99
    for r in rr:
        rec=float(r["recall"])
        if rec+1e-9>=best: pts.append(r); best=max(best,rec)
    return pts
def interp(rr,t,field):
    p=mono(rr)
    if t<float(p[0]["recall"]) or t>float(p[-1]["recall"]): return None
    for a,b in zip(p,p[1:]):
        ra,rb=float(a["recall"]),float(b["recall"])
        if ra<=t<=rb:
            x=0 if rb<=ra+1e-12 else (t-ra)/(rb-ra)
            return float(a[field])+x*(float(b[field])-float(a[field]))
    return float(p[-1][field])
def mean_match(ref,test):
    vals=[]
    for t in TARGETS:
        ri=interp(ref,t,"mean_ios"); ti=interp(test,t,"mean_ios")
        rl=interp(ref,t,"mean_latency"); tl=interp(test,t,"mean_latency")
        if None not in (ri,ti,rl,tl): vals.append((100*(1-ti/ri),100*(1-tl/rl)))
    return {"points":len(vals),
            "mean_io_saving_percent":float(np.mean([x[0] for x in vals])) if vals else None,
            "mean_latency_saving_percent":float(np.mean([x[1] for x in vals])) if vals else None}

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","heldout","gt5000","index-prefix","ivf","work","out"): ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    for n in ("binary","heldout","gt5000","index_prefix","ivf","work","out"): setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    result={"controller_blob":"2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb","window":WINDOW,"history":[]}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for h in HISTORY:
            if h+WINDOW>5000: continue
            qprefix=a.work/f"h{h}-prefix.fbin"; gprefix=a.work/f"h{h}-prefix.gt"
            qeval=a.work/f"h{h}-eval.fbin"; geval=a.work/f"h{h}-eval.gt"
            slice_fbin(a.heldout,qprefix,0,h+WINDOW); slice_gt(a.gt5000,gprefix,0,h+WINDOW)
            slice_fbin(a.heldout,qeval,h,WINDOW); slice_gt(a.gt5000,geval,h,WINDOW)
            ivf,_=run(a.binary,a.out,f"h{h}-ivf",qeval,geval,a.index_prefix,a.ivf,False,0,0)
            core,cst=run(a.binary,a.out,f"h{h}-core",qprefix,gprefix,a.index_prefix,a.ivf,True,0,h)
            full,fst=run(a.binary,a.out,f"h{h}-sample2",qprefix,gprefix,a.index_prefix,a.ivf,True,10,h)
            row={"history_queries":h,"measured_queries":WINDOW,
                 "core_vs_ivf":mean_match(ivf,core),"sample2_vs_ivf":mean_match(ivf,full),
                 "sample2_increment_over_core":mean_match(core,full),
                 "sample2_L160_stats":fst.get("160",{})}
            result["history"].append(row); save(a.out/"warmup.partial.json",result)
            print(json.dumps(row),flush=True)
    finally: os.sched_setaffinity(0,set(allowed))
    save(a.out/"final-warmup.json",result); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
