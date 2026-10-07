#!/usr/bin/env python3
"""Evaluate causal online start-hub winner learning from an empty state."""
from __future__ import annotations
import argparse,fcntl,json,os,re,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("vertex","onlinehub10")
STAT_RE=re.compile(
 r"ONLINE_HUB_STATS L=(\d+) active_hubs=(\d+) learned=(\d+) "
 r"duplicates=(\d+) full=(\d+) batch_writes=(\d+) dirty_pages=(\d+) "
 r"dirty_entries=(\d+) writes_if_final_flush=(\d+)"
)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+"\n")

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
            if not b:raise ValueError("truncated")
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

def run(binary,out,tag,q,gt,prefix,ivf,capacity):
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
    env["DISKANN_ONLINE_HUB_REPLAY"]="1"
    env["DISKANN_ONLINE_HUB_CAPACITY"]=str(capacity)
    env["DISKANN_ONLINE_HUB_WARMUP"]="4000"
    env["DISKANN_ONLINE_HUB_FLUSH_BATCH"]="10"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS):raise ValueError(f"{tag}: bad L")
    stats={}
    txt=log.read_text()
    for m in STAT_RE.finditer(txt):
        l=int(m.group(1))
        stats[str(l)]={
          "active_hubs":int(m.group(2)),
          "learned":int(m.group(3)),
          "duplicates":int(m.group(4)),
          "full":int(m.group(5)),
          "batch_writes":int(m.group(6)),
          "dirty_pages":int(m.group(7)),
          "dirty_entries":int(m.group(8)),
          "writes_if_final_flush":int(m.group(9)),
        }
    if len(stats)!=len(LS):
        raise ValueError(f"{tag}: expected {len(LS)} online-hub stat rows, got {len(stats)}")
    return rr,stats

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rs])),
        }
    return out

def aggregate_stats(reps):
    out={}
    for l in LS:
        a=[r[str(l)] for r in reps]
        out[str(l)]={k:float(np.mean([x[k] for x in a])) for k in a[0]}
        out[str(l)]["writes_per_replayed_query_if_final_flush"]=out[str(l)]["writes_if_final_flush"]/5000.0
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
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    q=args.work/"replay5000.fbin";range_fbin(args.queries,q,5000,5000)

    cfg={"vertex":0,"onlinehub10":10}
    runs={m:[] for m in METHODS};diag={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            order=list(METHODS[rep%2:]+METHODS[:rep%2])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                rr,st=run(args.binary,args.out,f"r{rep}-{m}",q,args.gt5000,args.index_prefix,args.ivf_16k,cfg[m])
                runs[m].append(rr);diag[m].append(st)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(f"{m} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.2f} lat={float(r40['mean_latency']):.1f}",flush=True)
    finally:os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}
    diagnostics={m:aggregate_stats(diag[m]) for m in METHODS}
    base=summary["vertex"];targets=(35.,45.,55.,65.,72.,75.);fixed={}
    for t in targets:
        bio=interp(base,t,"mean_ios");bl=interp(base,t,"latency_us")
        io=interp(summary["onlinehub10"],t,"mean_ios");la=interp(summary["onlinehub10"],t,"latency_us")
        if None in (bio,bl,io,la):continue
        fixed[str(t)]={
          "io_change_percent_vs_vertex":100*(io/bio-1),
          "latency_change_percent_vs_vertex":100*(la/bl-1),
        }

    vals=list(fixed.values())
    result={
      "policy":"causal online start-hub learning from empty state; each completed query contributes distinct rank1 winner to its selected 16K hub",
      "online_hub":{"capacity":10,"visibility":"immediate via write buffer","flush_batch":10,"physical_page_mutation":"counted, not executed"},
      "evaluation":{"replay_queries":5000,"warmup":4000,"measured":1000,"Ls":list(LS),"reps":args.reps},
      "summary":summary,
      "diagnostics":diagnostics,
      "fixed_recall":fixed,
      "mean_fixed_recall":{
        "io_change_percent_vs_vertex":float(np.mean([x["io_change_percent_vs_vertex"] for x in vals])),
        "latency_change_percent_vs_vertex":float(np.mean([x["latency_change_percent_vs_vertex"] for x in vals])),
      },
    }
    save(args.out/"online-hub-learning.json",result);print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
