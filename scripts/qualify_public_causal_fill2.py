#!/usr/bin/env python3
"""Evaluate the frozen causal Fill2 NavHints controller on a public ANN split."""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("baseline","ivf","cache","sample2")
CFG={
    "baseline":{"ivf":False,"experience":False,"cache":0,"hub":0,"sample":2,"fill":False},
    "ivf":{"ivf":True,"experience":False,"cache":0,"hub":0,"sample":2,"fill":False},
    "cache":{"ivf":True,"experience":True,"cache":512,"hub":0,"sample":2,"fill":False},
    "sample2":{"ivf":True,"experience":True,"cache":512,"hub":10,"sample":2,"fill":True},
}
STAT_KEYS=(
    "cache_capacity","hub_capacity","sample_denominator","fill",
    "cache_inserts","cache_skips","cache_evictions",
    "active_direct_hubs","persisted_hubs","direct_learned",
    "direct_duplicates","direct_full","sample_trials","sample_accepts","sample_rejects","fifo_evictions","writes","eval_writes",
    "write_slots","write_direct_slots","write_filler_slots",
    "eval_write_slots","eval_write_filler_slots","final_page_slots",
)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2)+"\n")

def xshape(p):
    with p.open("rb") as f:
        raw=f.read(8)
    if len(raw)!=8: raise ValueError(f"bad xbin header: {p}")
    return struct.unpack("<II",raw)

def slice_xbin(src,dst,start,count,itemsize):
    rows,dim=xshape(src)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad xbin slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*itemsize)
        fo.write(struct.pack("<II",count,dim))
        rem=count*dim*itemsize
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated xbin")
            fo.write(b); rem-=len(b)

def slice_gt(src,dst,start,count):
    with src.open("rb") as f:
        raw=f.read(8)
    rows,k=struct.unpack("<II",raw)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad GT slice")
    row_bytes=k*4
    ids_bytes=rows*row_bytes
    with src.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        fi.seek(8+start*row_bytes)
        rem=count*row_bytes
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated GT ids")
            fo.write(b); rem-=len(b)
        fi.seek(8+ids_bytes+start*row_bytes)
        rem=count*row_bytes
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated GT distances")
            fo.write(b); rem-=len(b)

def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o: out.append(o)
        else:
            for v in o.values(): out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(result_rows(v))
    return out

def parse_stats(log):
    out={}
    for line in log.read_text().splitlines():
        if "EXPERIENCE_STATS " not in line: continue
        kv={}
        for tok in line.split("EXPERIENCE_STATS ",1)[1].strip().split():
            if "=" in tok:
                k,v=tok.split("=",1); kv[k]=int(v)
        if "L" in kv:
            out[str(kv["L"])]={k:kv[k] for k in STAT_KEYS if k in kv}
    if set(out)!=set(map(str,LS)):
        raise ValueError(f"missing experience stats: {sorted(out)}")
    return out

def run(binary,out,tag,queries,gt,prefix,data_type,distance,ivf,cfg):
    job={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":data_type,"load_path":str(prefix)},
      "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":distance,"vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; op=out/f"{tag}.output.json"; log=out/f"{tag}.log"; save(inp,job)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"): env.pop(n,None)
    if cfg["ivf"]:
        env["DISKANN_HINT_IVF_FILE"]=str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"]="8"
        env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if cfg["experience"]:
        env["DISKANN_EXPERIENCE_REPLAY"]="1"
        env["DISKANN_EXPERIENCE_WARMUP"]="4000"
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]=str(cfg["cache"])
        env["DISKANN_EXPERIENCE_HUB_CAPACITY"]=str(cfg["hub"])
        env["DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR"]=str(cfg["sample"])
        env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="1" if cfg["fill"] else "0"
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS): raise ValueError(f"{tag}: bad L grid")
    return rr, (parse_stats(log) if cfg["experience"] else None)

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "qps":float(np.median([float(r["qps"]) for r in rs])),
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
        d["final_mean_page_occupancy"]=d["final_page_slots"]/d["persisted_hubs"] if d["persisted_hubs"] else 0.0
        out[str(l)]=d
    return out

def mono(s):
    out=[]; best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best:
            out.append((r,s[str(l)])); best=max(best,r)
    return out

def interp(s,t,f):
    p=mono(s)
    if not p or t<p[0][0] or t>p[-1][0]: return None
    for (a,ra),(b,rb) in zip(p,p[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[f])+x*(float(rb[f])-float(ra[f]))
    return float(p[-1][1][f])

def matched(summary):
    rows=[]
    for l in LS:
        target=summary["sample2"][str(l)]["recall_percent"]
        row={"sample2_L":l,"recall_percent":target}
        for ref in ("baseline","ivf","cache"):
            fio=summary["sample2"][str(l)]["mean_ios"]
            fla=summary["sample2"][str(l)]["latency_us"]
            rio=interp(summary[ref],target,"mean_ios")
            rla=interp(summary[ref],target,"latency_us")
            if rio is not None and rla is not None:
                row[ref]={
                  "io_saving_percent":100*(1-fio/rio),
                  "latency_saving_percent":100*(1-fla/rla),
                  "reference_ios":rio,
                  "reference_latency_us":rla,
                }
        rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","dataset-manifest","index-prefix","ivf","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","dataset_manifest","index_prefix","ivf","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    m=json.loads(args.dataset_manifest.read_text())
    replay=Path(m["files"]["heldout5000"])
    replay_gt=Path(m["files"]["heldout5000_gt"])
    itemsize={"float32":4,"uint8":1}[m["data_type"]]
    evalq=args.work/("eval1000"+replay.suffix)
    evalgt=args.work/"eval1000.gt"
    slice_xbin(replay,evalq,4000,1000,itemsize)
    slice_gt(replay_gt,evalgt,4000,1000)

    runs={x:[] for x in METHODS}; stats={x:[] for x in METHODS if CFG[x]["experience"]}
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS: raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS); order=list(METHODS[shift:]+METHODS[:shift])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for method in order:
                cfg=CFG[method]
                q=replay if cfg["experience"] else evalq
                gt=replay_gt if cfg["experience"] else evalgt
                if not cfg["experience"]:
                    warmq=args.work/f"warm4000{replay.suffix}"
                    warmgt=args.work/"warm4000.gt"
                    if not warmq.exists():
                        slice_xbin(replay,warmq,0,4000,itemsize)
                        slice_gt(replay_gt,warmgt,0,4000)
                    run(args.binary,args.out,f"r{rep}-{method}-warm",warmq,warmgt,args.index_prefix,
                        m["data_type"],m["metric"],args.ivf,cfg)
                rr,st=run(args.binary,args.out,f"r{rep}-{method}",q,gt,args.index_prefix,
                          m["data_type"],m["metric"],args.ivf,cfg)
                runs[method].append(rr)
                if st is not None: stats[method].append(st)
                r80=next(x for x in rr if int(x["search_l"])==80)
                print(f"{method} L80 recall={float(r80['recall']):.3f} io={float(r80['mean_ios']):.2f} lat={float(r80['mean_latency']):.1f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={mth:aggregate(runs[mth]) for mth in METHODS}
    write_stats={mth:aggregate_stats(stats[mth]) for mth in stats}
    match=matched(summary)
    means={}
    for ref in ("baseline","ivf","cache"):
        vals=[r[ref] for r in match if ref in r]
        means[ref]={
          "points":len(vals),
          "mean_io_saving_percent":float(np.mean([v["io_saving_percent"] for v in vals])) if vals else None,
          "mean_latency_saving_percent":float(np.mean([v["latency_saving_percent"] for v in vals])) if vals else None,
        }
    ivfm=json.loads(args.ivf.with_suffix(args.ivf.suffix+".manifest.json").read_text())
    result={
      "dataset":m["dataset"],"split":m["split"],"data_type":m["data_type"],"distance":m["metric"],
      "frozen_controller":{"entry_ids":16000,"nlist":512,"nprobe":8,"cache_ids":512,
        "hub_winners":10,"sample_denominator":2,"fill_from_cache":True,
        "offline_train_queries":5000,"online_warmup_queries":4000,"measured_queries":1000},
      "runtime_ivf_file_bytes":args.ivf.stat().st_size,
      "training":{"teacher_L":4,"landmarks":ivfm.get("landmark_ids"),"nlist":ivfm["nlist"]},
      "search":{"K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS},
      "summary":summary,"write_stats":write_stats,
      "matched_sample2":match,"mean_matched_sample2":means,
    }
    save(args.out/"public-causal-sample2.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
