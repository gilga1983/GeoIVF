#!/usr/bin/env python3
"""Equal-work demand control: random corpus IDs vs historical destinations vs Recent512."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("entry","randomdb512","history512","recent512")
CFG={
    "entry":       {"cache":0,   "mode":"fifo"},
    "randomdb512": {"cache":512, "mode":"randomdb"},
    "history512":  {"cache":512, "mode":"history"},
    "recent512":   {"cache":512, "mode":"fifo"},
}

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2)+"\n")

def shape(p):
    with p.open("rb") as f:return struct.unpack("<II",f.read(8))

def range_fbin(src,dst,start,count):
    rows,dim=shape(src)
    if start<0 or count<=0 or start+count>rows:raise ValueError("bad query range")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4);fo.write(struct.pack("<II",count,dim))
        rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b:raise ValueError("truncated fbin")
            fo.write(b);rem-=len(b)

def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o:out.append(o)
        else:
            for v in o.values():out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o:out.extend(result_rows(v))
    return out

def run(binary,out,tag,q,gt,prefix,ivf,cfg):
    job={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json";op=out/f"{tag}.output.json";log=out/f"{tag}.log";save(inp,job)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):env.pop(n,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    env["DISKANN_EXPERIENCE_REPLAY"]="1"
    env["DISKANN_EXPERIENCE_WARMUP"]="4000"
    env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]=str(cfg["cache"])
    env["DISKANN_EXPERIENCE_RECENT_MODE"]=cfg["mode"]
    env["DISKANN_EXPERIENCE_RANDOM_DB_SIZE"]="1000000"
    env["DISKANN_EXPERIENCE_HUB_CAPACITY"]="0"
    env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"]="2"
    env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="0"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS):raise ValueError(f"{tag}: bad L grid")
    return rr

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rs])),
          "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rs])),
        }
    return out

def mono(s):
    out=[];best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best:
            out.append((r,s[str(l)]));best=max(best,r)
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
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    q=args.work/"replay5000.fbin";range_fbin(args.queries,q,5000,5000)

    runs={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS);order=list(METHODS[shift:]+METHODS[:shift])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                rr=run(args.binary,args.out,f"r{rep}-{m}",q,args.gt5000,args.index_prefix,args.ivf_16k,CFG[m])
                runs[m].append(rr)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(f"{m} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.2f} "
                      f"lat={float(r40['mean_latency']):.1f} cpu={float(r40['mean_cpu_time']):.1f}",flush=True)
    finally:
      os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}
    targets=(35.,45.,55.,65.,72.,75.)
    matched={}
    for t in targets:
        base_io=interp(summary["entry"],t,"mean_ios")
        if base_io is None:continue
        row={}
        for m in METHODS:
            io=interp(summary[m],t,"mean_ios")
            cpu=interp(summary[m],t,"cpu_us")
            if io is None:continue
            row[m]={
              "mean_ios":io,
              "io_change_percent_vs_entry":100*(io/base_io-1),
              "cpu_us":cpu,
            }
        matched[str(t)]=row

    ranking=[]
    for m in METHODS[1:]:
        vals=[matched[str(t)][m]["io_change_percent_vs_entry"] for t in targets
              if str(t) in matched and m in matched[str(t)]]
        ranking.append({"method":m,"mean_io_change_percent_vs_entry":float(np.mean(vals)),
                        "targets":len(vals)})
    result={
      "question":"Does demand select better 512-ID navigation options than equal-work random controls?",
      "fairness":"All three controls scan 512 IDs with the same resident-PQ routine and offer only the best extra start; persistence is disabled.",
      "methods":{
        "randomdb512":"fixed deterministic 512 corpus IDs",
        "history512":"causal uniform reservoir over unique prior returned rank-1 destinations",
        "recent512":"causal FIFO/SkipDup last 512 unique returned rank-1 destinations",
      },
      "evaluation":{"Ls":list(LS),"reps":args.reps,"replay_queries":5000,"warmup":4000,"measured":1000},
      "summary":summary,
      "matched_recall":matched,
      "ranking":ranking,
    }
    save(args.out/"demand-vs-random.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
