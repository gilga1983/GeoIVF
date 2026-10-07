#!/usr/bin/env python3
"""Train-support distribution-shift stress test for frozen NavHints."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4; BEAM=8; K=10
LS=(10,20,40,80,160,320)
TARGETS=(35.,45.,55.,65.,72.,75.)
WARM=2000; MEASURE=500; KMEANS=32; SEED=20261007

def save(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2)+"\n")
def load_fbin(p):
    with Path(p).open("rb") as f: rows,dim=struct.unpack("<II",f.read(8))
    return np.fromfile(p,dtype="<f4",offset=8).reshape(rows,dim)
def save_fbin(p,a):
    with Path(p).open("wb") as f: f.write(struct.pack("<II",a.shape[0],a.shape[1])); a.astype("<f4",copy=False).tofile(f)
def load_gt(p):
    p=Path(p)
    with p.open("rb") as f: rows,k=struct.unpack("<II",f.read(8))
    ids=np.fromfile(p,dtype="<u4",count=rows*k,offset=8).reshape(rows,k)
    base=8+rows*k*4; d=None
    if p.stat().st_size==base*2-8: d=np.fromfile(p,dtype="<f4",count=rows*k,offset=base).reshape(rows,k)
    elif p.stat().st_size!=base: raise ValueError("unexpected gt size")
    return ids,d
def save_gt(p,ids,d):
    rows,k=ids.shape
    with Path(p).open("wb") as f:
        f.write(struct.pack("<II",rows,k)); ids.astype("<u4",copy=False).tofile(f)
        if d is not None: d.astype("<f4",copy=False).tofile(f)
def norm(a):
    n=np.linalg.norm(a,axis=1,keepdims=True); n[n==0]=1
    return a/n
def spherical_kmeans(a,k,iters=12):
    rng=np.random.default_rng(SEED); x=norm(a.astype(np.float32,copy=False))
    c=x[rng.choice(len(x),size=k,replace=False)].copy()
    for _ in range(iters):
        lab=np.argmax(x@c.T,axis=1); nc=np.zeros_like(c)
        for j in range(k):
            pts=x[lab==j]
            nc[j]=pts.mean(axis=0) if len(pts) else x[rng.integers(len(x))]
        c=norm(nc)
    return c
def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o: out.append(o)
        else:
            for v in o.values(): out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(result_rows(v))
    return out
def run(binary,out,tag,q,gt,prefix,ivf,mode,warmup):
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
    if mode!="ivf":
        env["DISKANN_EXPERIENCE_REPLAY"]="1"; env["DISKANN_EXPERIENCE_WARMUP"]=str(warmup)
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]="512"; env["DISKANN_EXPERIENCE_HUB_CAPACITY"]="0" if mode=="core" else "10"
        env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"]="2"; env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="1"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    return sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
def mono(rr):
    out=[];best=-1e99
    for r in rr:
        x=float(r["recall"])
        if x+1e-9>=best: out.append(r);best=max(best,x)
    return out
def interp(rr,t,f):
    p=mono(rr)
    if t<float(p[0]["recall"]) or t>float(p[-1]["recall"]): return None
    for a,b in zip(p,p[1:]):
        ra,rb=float(a["recall"]),float(b["recall"])
        if ra<=t<=rb:
            z=0 if rb<=ra+1e-12 else (t-ra)/(rb-ra)
            return float(a[f])+z*(float(b[f])-float(a[f]))
    return float(p[-1][f])
def compare(ref,test):
    vals=[]
    for t in TARGETS:
        ri=interp(ref,t,"mean_ios");ti=interp(test,t,"mean_ios")
        rl=interp(ref,t,"mean_latency");tl=interp(test,t,"mean_latency")
        if None not in (ri,ti,rl,tl): vals.append((100*(1-ti/ri),100*(1-tl/rl)))
    return {"points":len(vals),"io_saving_percent":float(np.mean([x[0] for x in vals])) if vals else None,
            "latency_saving_percent":float(np.mean([x[1] for x in vals])) if vals else None}

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries10k","gt5000","index-prefix","ivf","work","out"): ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    for n in ("binary","queries10k","gt5000","index_prefix","ivf","work","out"): setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    q=load_fbin(a.queries10k); gids,gd=load_gt(a.gt5000)
    if len(q)!=10000 or len(gids)!=5000: raise ValueError("unexpected split")
    train=q[:5000]; held=q[5000:]
    cent=spherical_kmeans(train,KMEANS)
    support=np.max(norm(held)@cent.T,axis=1)
    order=np.argsort(support)
    groups={"far":np.sort(order[:2500]),"near":np.sort(order[-2500:])}
    result={"controller_blob":"2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb",
            "definition":"heldout support = maximum cosine similarity to 32 spherical centroids fit on static-training queries",
            "warmup":WARM,"measured":MEASURE,"groups":[]}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,idx in groups.items():
            idx=idx[:WARM+MEASURE]
            qs=held[idx]; gi=gids[idx]; gd2=None if gd is None else gd[idx]
            qp=a.work/f"{name}.fbin";gp=a.work/f"{name}.gt";save_fbin(qp,qs);save_gt(gp,gi,gd2)
            qw=a.work/f"{name}-warm.fbin";gw=a.work/f"{name}-warm.gt";save_fbin(qw,qs[:WARM]);save_gt(gw,gi[:WARM],None if gd2 is None else gd2[:WARM])
            qe=a.work/f"{name}-eval.fbin";ge=a.work/f"{name}-eval.gt";save_fbin(qe,qs[WARM:]);save_gt(ge,gi[WARM:],None if gd2 is None else gd2[WARM:])
            run(a.binary,a.out,f"{name}-ivf-cachewarm",qw,gw,a.index_prefix,a.ivf,"ivf",0)
            ivf=run(a.binary,a.out,f"{name}-ivf",qe,ge,a.index_prefix,a.ivf,"ivf",0)
            core=run(a.binary,a.out,f"{name}-core",qp,gp,a.index_prefix,a.ivf,"core",WARM)
            full=run(a.binary,a.out,f"{name}-sample2",qp,gp,a.index_prefix,a.ivf,"sample2",WARM)
            row={"group":name,"queries":len(idx),"mean_support_similarity":float(np.mean(support[idx])),
                 "core_vs_ivf":compare(ivf,core),"sample2_vs_ivf":compare(ivf,full),
                 "sample2_increment_over_core":compare(core,full)}
            result["groups"].append(row);save(a.out/"shift.partial.json",result);print(json.dumps(row),flush=True)
    finally: os.sched_setaffinity(0,set(allowed))
    save(a.out/"final-shift.json",result);print(json.dumps(result,indent=2))

if __name__=="__main__":main()
