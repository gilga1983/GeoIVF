#!/usr/bin/env python3
"""Screen consolidated experience controller and cache-packed hub writes."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("ivf","cache","nofill4","fill1","fill2","fill4","fill10")
CFG={
    "ivf":     {"cache":0,   "hub":0,  "threshold":4,  "fill":False},
    "cache":   {"cache":512, "hub":0,  "threshold":4,  "fill":False},
    "nofill4": {"cache":512, "hub":10, "threshold":4,  "fill":False},
    "fill1":   {"cache":512, "hub":10, "threshold":1,  "fill":True},
    "fill2":   {"cache":512, "hub":10, "threshold":2,  "fill":True},
    "fill4":   {"cache":512, "hub":10, "threshold":4,  "fill":True},
    "fill10":  {"cache":512, "hub":10, "threshold":10, "fill":True},
}
STAT_KEYS=(
    "cache_capacity","hub_capacity","flush_threshold","fill",
    "cache_inserts","cache_skips","cache_evictions",
    "active_direct_hubs","persisted_hubs","direct_learned",
    "direct_duplicates","direct_full","fifo_evictions","writes","eval_writes",
    "write_slots","write_direct_slots","write_filler_slots",
    "eval_write_slots","eval_write_filler_slots","final_page_slots",
    "pending_hubs","pending_entries",
)

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

def parse_stats(log):
    out={}
    for line in log.read_text().splitlines():
        if "EXPERIENCE_STATS " not in line:continue
        payload=line.split("EXPERIENCE_STATS ",1)[1].strip()
        kv={}
        for tok in payload.split():
            if "=" not in tok:continue
            k,v=tok.split("=",1);kv[k]=int(v)
        if "L" in kv:
            out[str(kv["L"])]={k:kv[k] for k in STAT_KEYS if k in kv}
    if set(out)!=set(map(str,LS)):
        raise ValueError(f"missing experience stats: {sorted(out)}")
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
    env["DISKANN_EXPERIENCE_HUB_CAPACITY"]=str(cfg["hub"])
    env["DISKANN_EXPERIENCE_FLUSH_THRESHOLD"]=str(cfg["threshold"])
    env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="1" if cfg["fill"] else "0"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS):raise ValueError(f"{tag}: bad L grid")
    return rr,parse_stats(log)

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

def aggregate_stats(reps):
    out={}
    for l in LS:
        rows=[x[str(l)] for x in reps]
        d={k:float(np.mean([r[k] for r in rows])) for k in STAT_KEYS}
        d["writes_per_query"]=d["writes"]/5000.0
        d["eval_writes_per_query"]=d["eval_writes"]/1000.0
        d["mean_write_occupancy"]=d["write_slots"]/d["writes"] if d["writes"] else 0.0
        d["mean_filler_slots_per_write"]=d["write_filler_slots"]/d["writes"] if d["writes"] else 0.0
        d["eval_mean_write_occupancy"]=d["eval_write_slots"]/d["eval_writes"] if d["eval_writes"] else 0.0
        d["eval_mean_filler_slots_per_write"]=d["eval_write_filler_slots"]/d["eval_writes"] if d["eval_writes"] else 0.0
        d["final_mean_page_occupancy"]=d["final_page_slots"]/d["persisted_hubs"] if d["persisted_hubs"] else 0.0
        out[str(l)]=d
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

def pair(summary,a,b,targets):
    rows=[]
    for t in targets:
        aio=interp(summary[a],t,"mean_ios");bio=interp(summary[b],t,"mean_ios")
        ala=interp(summary[a],t,"latency_us");bla=interp(summary[b],t,"latency_us")
        if None in (aio,bio,ala,bla):continue
        rows.append({
          "recall":t,
          "io_change_percent":100*(bio/aio-1),
          "latency_change_percent":100*(bla/ala-1),
        })
    return rows

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

    runs={m:[] for m in METHODS};stats={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS);order=list(METHODS[shift:]+METHODS[:shift])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                rr,st=run(args.binary,args.out,f"r{rep}-{m}",q,args.gt5000,args.index_prefix,args.ivf_16k,CFG[m])
                runs[m].append(rr);stats[m].append(st)
                r40=next(x for x in rr if int(x["search_l"])==40)
                s40=st["40"]
                occ=s40["write_slots"]/s40["writes"] if s40["writes"] else 0.0
                print(
                  f"{m} L40 recall={float(r40['recall']):.3f} io={float(r40['mean_ios']):.2f} "
                  f"lat={float(r40['mean_latency']):.1f} writes={s40['writes']} occ={occ:.2f}",
                  flush=True)
    finally:os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}
    write_stats={m:aggregate_stats(stats[m]) for m in METHODS}
    targets=(35.,45.,55.,65.,72.,75.)
    fixed={}
    base=summary["ivf"]
    for t in targets:
        bio=interp(base,t,"mean_ios");bla=interp(base,t,"latency_us")
        if bio is None:continue
        fixed[str(t)]={}
        for m in METHODS:
            io=interp(summary[m],t,"mean_ios");la=interp(summary[m],t,"latency_us")
            if io is None:continue
            fixed[str(t)][m]={
              "io_change_percent_vs_ivf":100*(io/bio-1),
              "latency_change_percent_vs_ivf":100*(la/bla-1),
            }

    hub_increment={m:pair(summary,"cache",m,targets) for m in METHODS if m not in ("ivf","cache")}
    fill4_vs_nofill4=pair(summary,"nofill4","fill4",targets)

    ranking=[]
    for m in METHODS:
        if m=="ivf":continue
        vals=[fixed[str(t)][m] for t in targets if str(t) in fixed and m in fixed[str(t)]]
        ws=write_stats[m]["160"]
        ranking.append({
          "method":m,
          "mean_io_change_percent_vs_ivf":float(np.mean([x["io_change_percent_vs_ivf"] for x in vals])),
          "mean_latency_change_percent_vs_ivf":float(np.mean([x["latency_change_percent_vs_ivf"] for x in vals])),
          "L160_writes_per_query":ws["writes_per_query"],
          "L160_eval_writes_per_query":ws["eval_writes_per_query"],
          "L160_mean_write_occupancy":ws["mean_write_occupancy"],
          "L160_mean_filler_slots_per_write":ws["mean_filler_slots_per_write"],
          "L160_final_mean_page_occupancy":ws["final_mean_page_occupancy"],
        })
    ranking.sort(key=lambda x:x["mean_io_change_percent_vs_ivf"])

    result={
      "question":"What flush cadence best turns the 512-result cache into packed persistent 16K-hub shortcuts?",
      "architecture":"16K learned entry + online seed-only cache + causal persisted hub-page winners; no support2 or regional routing",
      "write_policy":"protect direct hub winners; on rewrite, fill spare slots with already-ranked top cache candidates for the triggering query; later direct winners replace filler",
      "methods":CFG,
      "evaluation":{"Ls":list(LS),"reps":args.reps,"replay_queries":5000,"warmup":4000,"measured":1000},
      "summary":summary,
      "write_stats":write_stats,
      "fixed_recall_vs_16k":fixed,
      "hub_increment_over_cache":hub_increment,
      "fill4_vs_nofill4":fill4_vs_nofill4,
      "ranking":ranking,
    }
    save(args.out/"experience-write-sweep.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
