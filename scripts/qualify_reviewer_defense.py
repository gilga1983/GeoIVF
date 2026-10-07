#!/usr/bin/env python3
"""Reviewer-defense diagnostics for the frozen NavHints core.

1) Quantify exact versus semantic repetition in the real heldout MedRAG trace.
2) Sweep the recent-result navigation-cache capacity under the frozen protocol.
3) If exact causal repeats exist, re-evaluate after removing them.
"""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
CAPS=(0,64,128,256,512,1024,2048)
TARGETS=(35.,45.,55.,65.,72.,75.)
REPS=3

def save(p,o):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(o,indent=2)+"\n")

def load_fbin(p):
    p=Path(p)
    with p.open("rb") as f:
        raw=f.read(8)
    if len(raw)!=8: raise ValueError("bad fbin header")
    rows,dim=struct.unpack("<II",raw)
    a=np.fromfile(p,dtype="<f4",offset=8)
    if a.size!=rows*dim: raise ValueError("bad fbin size")
    return a.reshape(rows,dim)

def save_fbin(p,a):
    a=np.asarray(a,dtype=np.float32)
    with Path(p).open("wb") as f:
        f.write(struct.pack("<II",a.shape[0],a.shape[1]))
        a.astype("<f4",copy=False).tofile(f)

def load_gt(p):
    p=Path(p)
    with p.open("rb") as f:
        raw=f.read(8)
    if len(raw)!=8: raise ValueError("bad gt header")
    rows,k=struct.unpack("<II",raw)
    ids_bytes=rows*k*4
    size=p.stat().st_size
    ids=np.fromfile(p,dtype="<u4",count=rows*k,offset=8).reshape(rows,k)
    d=None
    if size==8+2*ids_bytes:
        d=np.fromfile(p,dtype="<f4",count=rows*k,offset=8+ids_bytes).reshape(rows,k)
    elif size!=8+ids_bytes:
        raise ValueError(f"unexpected gt size {size}")
    return ids,d

def save_gt(p,ids,d=None):
    ids=np.asarray(ids,dtype=np.uint32)
    with Path(p).open("wb") as f:
        f.write(struct.pack("<II",ids.shape[0],ids.shape[1]))
        ids.astype("<u4",copy=False).tofile(f)
        if d is not None:
            np.asarray(d,dtype=np.float32).astype("<f4",copy=False).tofile(f)

def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o:
            out.append(o)
        else:
            for v in o.values(): out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(result_rows(v))
    return out

def config(out,q,gt,prefix):
    return {"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}

def run(binary,out,tag,q,gt,prefix,ivf,cap=None,warmup=0,skip_recall=False):
    inp=out/f"{tag}.input.json"; op=out/f"{tag}.output.json"; log=out/f"{tag}.log"
    save(inp,config(out,q,gt,prefix))
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"): env.pop(n,None)
    if skip_recall: env["DISKANN_SKIP_RECALL"]="1"
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if cap is not None:
        env["DISKANN_EXPERIENCE_REPLAY"]="1"
        env["DISKANN_EXPERIENCE_WARMUP"]=str(warmup)
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]=str(cap)
        env["DISKANN_EXPERIENCE_HUB_CAPACITY"]="0"
        env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"]="2"
        env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="0"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
                       stdout=lf,stderr=subprocess.STDOUT,env=env,check=True,timeout=2400)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rr]!=list(LS):
        raise ValueError(f"{tag}: bad L rows")
    return rr

def aggregate(reps):
    out={}
    for l in LS:
        rs=[next(r for r in rr if int(r["search_l"])==l) for rr in reps]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "cpu_us":float(np.median([float(r.get("mean_cpu_time",0)) for r in rs])),
          "qps":float(np.median([float(r["qps"]) for r in rs])),
        }
    return out

def monotone(s):
    out=[]; best=-1e99
    for l in LS:
        r=s[str(l)]; rec=float(r["recall_percent"])
        if rec+1e-9>=best:
            out.append((rec,r)); best=max(best,rec)
    return out

def interp(s,t,field):
    pts=monotone(s)
    if not pts or t<pts[0][0] or t>pts[-1][0]: return None
    for (a,ra),(b,rb) in zip(pts,pts[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[field])+x*(float(rb[field])-float(ra[field]))
    return float(pts[-1][1][field])

def compare(ref,test):
    vals=[]
    for t in TARGETS:
        ri=interp(ref,t,"mean_ios"); ti=interp(test,t,"mean_ios")
        rl=interp(ref,t,"latency_us"); tl=interp(test,t,"latency_us")
        rc=interp(ref,t,"cpu_us"); tc=interp(test,t,"cpu_us")
        if None not in (ri,ti,rl,tl):
            vals.append({
              "target":t,
              "io_saving_percent":100*(1-ti/ri),
              "latency_saving_percent":100*(1-tl/rl),
              "cpu_delta_us":None if rc is None or tc is None else tc-rc,
            })
    return {
      "points":len(vals),
      "mean_io_saving_percent":float(np.mean([x["io_saving_percent"] for x in vals])) if vals else None,
      "mean_latency_saving_percent":float(np.mean([x["latency_saving_percent"] for x in vals])) if vals else None,
      "mean_cpu_delta_us":float(np.mean([x["cpu_delta_us"] for x in vals if x["cpu_delta_us"] is not None])) if vals else None,
      "per_target":vals,
    }

def exact_repeat_diagnostics(q,gt):
    warm=q[:4000]; ev=q[4000:]
    # Exact causal repeats: a measured vector has appeared anywhere earlier in the heldout stream.
    seen={row.tobytes() for row in warm}
    causal_repeat=np.zeros(len(ev),dtype=bool)
    for i,row in enumerate(ev):
        key=row.tobytes()
        causal_repeat[i]=key in seen
        seen.add(key)

    all_keys=[row.tobytes() for row in q]
    unique=len(set(all_keys))

    # Semantic-nearest warm query, then neighbor-set overlap.
    wn=warm.astype(np.float32,copy=True); en=ev.astype(np.float32,copy=True)
    wn/=np.maximum(np.linalg.norm(wn,axis=1,keepdims=True),1e-12)
    en/=np.maximum(np.linalg.norm(en,axis=1,keepdims=True),1e-12)
    sims=en @ wn.T
    nearest=np.argmax(sims,axis=1)
    best=sims[np.arange(len(ev)),nearest]
    overlaps=np.empty(len(ev),dtype=np.float32)
    for i,j in enumerate(nearest):
        overlaps[i]=len(set(map(int,gt[4000+i,:10])) & set(map(int,gt[int(j),:10])))/10.0

    def qs(a):
        return {str(p):float(np.quantile(a,p)) for p in (0.0,0.1,0.25,0.5,0.75,0.9,0.99,1.0)}

    return {
      "heldout_queries":len(q),
      "unique_exact_vectors":unique,
      "exact_duplicate_vectors":len(q)-unique,
      "exact_duplicate_fraction":float((len(q)-unique)/len(q)),
      "measured_queries":len(ev),
      "measured_causal_exact_repeats":int(causal_repeat.sum()),
      "measured_causal_exact_repeat_fraction":float(causal_repeat.mean()),
      "nearest_prior_cosine_quantiles":qs(best),
      "nearest_prior_top10_overlap_quantiles":qs(overlaps),
      "mean_nearest_prior_top10_overlap":float(overlaps.mean()),
      "nonexact_eval_mask":(~causal_repeat).tolist(),
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","heldout","gt5000","index-prefix","ivf","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    for n in ("binary","heldout","gt5000","index_prefix","ivf","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)

    q=load_fbin(a.heldout); gids,gd=load_gt(a.gt5000)
    if len(q)!=5000 or len(gids)!=5000: raise ValueError("expected 5K heldout")
    diag=exact_repeat_diagnostics(q,gids)
    mask=np.array(diag.pop("nonexact_eval_mask"),dtype=bool)
    save(a.out/"trace-locality.json",diag)
    print(json.dumps(diag,indent=2),flush=True)

    warmq=a.work/"warm4000.fbin"; warmgt=a.work/"warm4000.gt"
    evalq=a.work/"eval1000.fbin"; evalgt=a.work/"eval1000.gt"
    save_fbin(warmq,q[:4000]); save_gt(warmgt,gids[:4000],None if gd is None else gd[:4000])
    save_fbin(evalq,q[4000:]); save_gt(evalgt,gids[4000:],None if gd is None else gd[4000:])

    methods=[f"cap{c}" for c in CAPS]
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(REPS):
            order=methods[rep:]+methods[:rep]
            for m in order:
                cap=int(m[3:])
                if cap==0:
                    run(a.binary,a.out,f"r{rep}-{m}-cachewarm",warmq,warmgt,a.index_prefix,a.ivf,
                        cap=None,warmup=0,skip_recall=True)
                    rr=run(a.binary,a.out,f"r{rep}-{m}",evalq,evalgt,a.index_prefix,a.ivf,
                           cap=None,warmup=0)
                else:
                    rr=run(a.binary,a.out,f"r{rep}-{m}",a.heldout,a.gt5000,a.index_prefix,a.ivf,
                           cap=cap,warmup=4000)
                runs[m].append(rr)
                save(a.out/"capacity-runs.partial.json",runs)
    finally:
      os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(v) for m,v in runs.items()}
    base=summary["cap0"]
    caprows=[]
    for cap in CAPS:
        m=f"cap{cap}"
        caprows.append({
          "capacity_ids":cap,
          "id_payload_bytes":4*cap,
          "vs_entry":compare(base,summary[m]),
        })

    nonexact=None
    if int(diag["measured_causal_exact_repeats"])>0 and int(mask.sum())>=100:
        nq=q[4000:][mask]; ng=gids[4000:][mask]; nd=None if gd is None else gd[4000:][mask]
        replayq=a.work/"nonexact-replay.fbin"; replaygt=a.work/"nonexact-replay.gt"
        nevalq=a.work/"nonexact-eval.fbin"; nevalgt=a.work/"nonexact-eval.gt"
        save_fbin(replayq,np.concatenate([q[:4000],nq],axis=0))
        save_gt(replaygt,np.concatenate([gids[:4000],ng],axis=0),
                None if gd is None else np.concatenate([gd[:4000],nd],axis=0))
        save_fbin(nevalq,nq); save_gt(nevalgt,ng,nd)
        run(a.binary,a.out,"nonexact-entry-cachewarm",warmq,warmgt,a.index_prefix,a.ivf,
            cap=None,warmup=0,skip_recall=True)
        br=run(a.binary,a.out,"nonexact-entry",nevalq,nevalgt,a.index_prefix,a.ivf,cap=None,warmup=0)
        cr=run(a.binary,a.out,"nonexact-core512",replayq,replaygt,a.index_prefix,a.ivf,cap=512,warmup=4000)
        nonexact={"measured_queries":int(mask.sum()),"core512_vs_entry":compare(aggregate([br]),aggregate([cr]))}

    result={
      "controller_blob":"2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb",
      "trace_diagnostics":diag,
      "capacity_sweep":caprows,
      "summary":summary,
      "nonexact_only":nonexact,
    }
    save(a.out/"reviewer-defense.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
