#!/usr/bin/env python3
"""Evaluate independent successful winners aggregated at deployed 16K hubs."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
COUNTS=(1,2,4,8,10)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+"\n")

def shape(p):
    with p.open("rb") as f:return struct.unpack("<II",f.read(8))

def suffix_fbin(src,dst,start):
    r,d=shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*d*4);fo.write(struct.pack("<II",r-start,d))
        rem=(r-start)*d*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b:raise ValueError("truncated")
            fo.write(b);rem-=len(b)

def suffix_gt(src,dst,start):
    raw=src.read_bytes();r,k=struct.unpack("<II",raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II",r-start,k));f.write(raw[8+start*k*4:])

def rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o:out.append(o)
        else:
            for v in o.values():out.extend(rows(v))
    elif isinstance(o,list):
        for v in o:out.extend(rows(v))
    return out

def run(binary,out,tag,q,gt,prefix,ivf,count):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json";op=out/f"{tag}.output.json";log=out/f"{tag}.log";save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):env.pop(n,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    env["DISKANN_VERTEX_HINT_VARIANT"]="1"
    if count:
        env["DISKANN_HUB_WINNER_COUNT"]=str(count)
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS):raise ValueError("bad L")
    return rr

def agg(reps):
    out={}
    for l in LS:
        a=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in a])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in a])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in a])),
          "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in a])),
          "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in a])),
        }
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
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","payload-manifest","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=5)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","payload_manifest","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    q=args.work/"held.fbin";gt=args.work/"held.gt";suffix_fbin(args.queries,q,9000);suffix_gt(args.gt5000,gt,4000)
    meta=json.loads(args.payload_manifest.read_text())

    methods=["vertex"]+[f"hub{k}" for k in COUNTS]
    runs={m:[] for m in methods}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(methods);order=methods[shift:]+methods[:shift]
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                count=0 if m=="vertex" else int(m[3:])
                rr=run(args.binary,args.out,f"r{rep}-{m}",q,gt,args.index_prefix,args.ivf_16k,count)
                runs[m].append(rr)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(f"{m} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.2f} lat={float(r40['mean_latency']):.1f}",flush=True)
    finally:os.sched_setaffinity(0,set(allowed))

    summary={m:agg(runs[m]) for m in methods};base=summary["vertex"]
    fixed={}
    for t in (35.,45.,55.,65.,72.,75.):
        bio=interp(base,t,"mean_ios");bl=interp(base,t,"latency_us")
        if bio is None:continue
        fixed[str(t)]={}
        for m in methods:
            io=interp(summary[m],t,"mean_ios");la=interp(summary[m],t,"latency_us")
            if io is None:continue
            fixed[str(t)][m]={
              "io_change_percent_vs_vertex":100*(io/bio-1),
              "latency_change_percent_vs_vertex":100*(la/bl-1),
            }
    ranking=[]
    for m in methods[1:]:
        vals=[fixed[str(t)][m] for t in (35.,45.,55.,65.,72.,75.) if str(t) in fixed and m in fixed[str(t)]]
        ranking.append({"method":m,
          "mean_io_change_percent":float(np.mean([x["io_change_percent_vs_vertex"] for x in vals])),
          "mean_latency_change_percent":float(np.mean([x["latency_change_percent_vs_vertex"] for x in vals]))})
    ranking.sort(key=lambda x:x["mean_io_change_percent"])
    result={"policy":"deployed 16K hub -> independent actual L160 winners from different warm searches",
      "payload":meta,"evaluation":{"Ls":list(LS),"reps":args.reps},"summary":summary,"fixed_recall":fixed,"ranking":ranking}
    save(args.out/"hub-winners.json",result);print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
