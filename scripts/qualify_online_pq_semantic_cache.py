#!/usr/bin/env python3
"""Evaluate real online PQ-key semantic cache with page-piggybacked results."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=1
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("vertex","c128","c512","c2048")
REPLAY_START=7000
WARMUP=2000
MEASURED=1000

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+"\n")
def fbin_shape(p):
    with p.open("rb") as f:r,d=struct.unpack("<II",f.read(8))
    return r,d
def slice_fbin(src,dst,start):
    r,d=fbin_shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*d*4);fo.write(struct.pack("<II",r-start,d))
        rem=(r-start)*d*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b:raise ValueError("truncated")
            fo.write(b);rem-=len(b)
def slice_gt(src,dst,start):
    raw=src.read_bytes();r,k=struct.unpack("<II",raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II",r-start,k));f.write(raw[8+start*k*4:])
def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o:out.append(o)
        else:
            for v in o.values():out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o:out.extend(result_rows(v))
    return out
def cap(method):
    return 0 if method=="vertex" else int(method[1:])
def run_one(binary,out,tag,queries,gt,index_prefix,ivf,codes,method):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(index_prefix)},
      "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
      "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
      "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
      "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json";output=out/f"{tag}.output.json";log=out/f"{tag}.log";save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):env.pop(n,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf);env["DISKANN_HINT_IVF_NPROBE"]="8";env["DISKANN_HINT_IVF_MAX_STARTS"]="1";env["DISKANN_VERTEX_HINT_VARIANT"]="1"
    c=cap(method)
    if c:
        env["DISKANN_SEMANTIC_CACHE_CAPACITY"]=str(c)
        env["DISKANN_SEMANTIC_CACHE_RESULTS"]="10"
        env["DISKANN_SEMANTIC_CACHE_WARMUP"]=str(WARMUP)
        env["DISKANN_SEMANTIC_QUERY_CODES_FILE"]=str(codes)
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(output)],stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS):raise ValueError("bad L")
    return rows
def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={"recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
                     "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
                     "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
                     "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rs])),
                     "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rs]))}
    return out
def mono(s):
    out=[];best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best:out.append((r,s[str(l)]));best=max(best,r)
    return out
def interp(s,t,f):
    p=mono(s)
    if t<p[0][0] or t>p[-1][0]:return None
    for (a,ra),(b,rb) in zip(p,p[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[f])+x*(float(rb[f])-float(ra[f]))
    return float(p[-1][1][f])
def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","query-codes","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","query_codes","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    replay_q=args.work/"replay3000.fbin";replay_gt=args.work/"replay3000.gt";eval_q=args.work/"eval1000.fbin";eval_gt=args.work/"eval1000.gt"
    slice_fbin(args.queries,replay_q,REPLAY_START);slice_gt(args.gt5000,replay_gt,REPLAY_START-5000)
    slice_fbin(args.queries,eval_q,9000);slice_gt(args.gt5000,eval_gt,4000)
    if fbin_shape(replay_q)[0]!=3000 or fbin_shape(eval_q)[0]!=1000:raise ValueError("slice mismatch")
    runs={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                q=eval_q if m=="vertex" else replay_q
                gt=eval_gt if m=="vertex" else replay_gt
                rr=run_one(args.binary,args.out,f"r{rep}-{m}",q,gt,args.index_prefix,args.ivf_16k,args.query_codes,m);runs[m].append(rr)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(f"{m} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.2f} lat={float(r40['mean_latency']):.1f} cpu={float(r40['mean_cpu_time']):.1f}",flush=True)
    finally:os.sched_setaffinity(0,set(allowed))
    summary={m:aggregate(runs[m]) for m in METHODS};base=summary["vertex"]
    fixed={}
    for t in (35.,45.,55.,65.,72.,75.):
        bio=interp(base,t,"mean_ios");blat=interp(base,t,"latency_us");bcpu=interp(base,t,"cpu_us")
        if bio is None:continue
        fixed[str(t)]={}
        for m in METHODS:
            io=interp(summary[m],t,"mean_ios");lat=interp(summary[m],t,"latency_us");cpu=interp(summary[m],t,"cpu_us")
            if io is None:continue
            fixed[str(t)][m]={"io_change_percent":100*(io/bio-1),"latency_change_percent":100*(lat/blat-1),"cpu_change_percent":100*(cpu/bcpu-1)}
    ranking=[]
    for m in METHODS[1:]:
        c=cap(m);vals=[fixed[t][m] for t in fixed if m in fixed[t]]
        directory=c*(64+4);with_siblings=c*(64+4+9*4)
        ranking.append({"method":m,"capacity":c,"directory_kib":directory/1024,"directory_plus_siblings_kib":with_siblings/1024,
                        "mean_io_change_percent":float(np.mean([x["io_change_percent"] for x in vals])),
                        "mean_latency_change_percent":float(np.mean([x["latency_change_percent"] for x in vals])),
                        "mean_cpu_change_percent":float(np.mean([x["cpu_change_percent"] for x in vals]))})
    ranking.sort(key=lambda x:x["mean_latency_change_percent"])
    result={"policy":"actual online PQ-key rolling semantic cache; actual prior DiskANN outputs; one anchor start; siblings exposed only after anchor expansion; support-2 vertex hints",
            "replay_start":REPLAY_START,"warmup_queries":WARMUP,"measured_queries":MEASURED,"threads":THREADS,
            "summary":summary,"fixed_recall":fixed,"ranking":ranking}
    save(args.out/"online-pq-semantic-cache.json",result);print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
