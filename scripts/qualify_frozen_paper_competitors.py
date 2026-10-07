#!/usr/bin/env python3
"""Same-device competitor frontier for the frozen causal Fill2 controller."""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,12,20,22,24,40,44,80,88,160,176,320)
NAV_ANCHORS=(10,20,40,80,160,320)
CACHE_COUNTS=(30,32)
DIM=768
FLOAT_BYTES=4
MAX_DEGREE=64
STAT_KEYS=(
 "cache_capacity","hub_capacity","flush_threshold","fill","cache_inserts","cache_skips",
 "cache_evictions","active_direct_hubs","persisted_hubs","direct_learned",
 "direct_duplicates","direct_full","fifo_evictions","writes","eval_writes","write_slots",
 "write_direct_slots","write_filler_slots","eval_write_slots","eval_write_filler_slots",
 "final_page_slots","pending_hubs","pending_entries",
)

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2)+"\n")

def fshape(p):
    with p.open("rb") as f: raw=f.read(8)
    rows,dim=struct.unpack("<II",raw)
    if p.stat().st_size!=8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def slice_fbin(src,dst,start,count):
    rows,dim=fshape(src)
    if start+count>rows: raise ValueError("bad query slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4); fo.write(struct.pack("<II",count,dim))
        rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)

def slice_gt(src,dst,start,count):
    with src.open("rb") as f: rows,k=struct.unpack("<II",f.read(8))
    if start+count>rows: raise ValueError("bad gt slice")
    rb=k*4; ids_bytes=rows*rb
    with src.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        fi.seek(8+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated gt ids")
            fo.write(b); rem-=len(b)
        fi.seek(8+ids_bytes+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated gt dists")
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
        if "L" in kv: out[str(kv["L"])]={k:kv[k] for k in STAT_KEYS if k in kv}
    if set(out)!=set(map(str,LS)): raise ValueError(f"missing experience stats {sorted(out)}")
    return out

def run_one(binary,out,tag,queries,gt,index_prefix,*,cache_nodes=None,hot_ids=None,qsev=None,ivf=None,experience=False):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(index_prefix)},
      "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
       "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
       "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":cache_nodes,
       "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; op=out/f"{tag}.output.json"; log=out/f"{tag}.log"; save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"): env.pop(n,None)
    if hot_ids is not None: env["DISKANN_STATIC_CACHE_IDS_FILE"]=str(hot_ids)
    if qsev is not None: env["DISKANN_QSEV_FILE"]=str(qsev)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"]=str(ivf); env["DISKANN_HINT_IVF_NPROBE"]="8"; env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if experience:
        env["DISKANN_EXPERIENCE_REPLAY"]="1"; env["DISKANN_EXPERIENCE_WARMUP"]="4000"
        env["DISKANN_EXPERIENCE_CACHE_CAPACITY"]="512"; env["DISKANN_EXPERIENCE_HUB_CAPACITY"]="10"
        env["DISKANN_EXPERIENCE_FLUSH_THRESHOLD"]="2"; env["DISKANN_EXPERIENCE_FILL_FROM_CACHE"]="1"
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(op)]
    t=time.monotonic()
    with log.open("w") as lf:
        subprocess.run(cmd,stdout=lf,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
    print(f"completed {tag} in {time.monotonic()-t:.1f}s",flush=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS): raise ValueError(f"{tag}: bad L grid")
    return rr,(parse_stats(log) if experience else None)

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "rounds":len(rs),"recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "median_qps":float(np.median([float(r["qps"]) for r in rs])),
          "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "median_io_us":float(np.median([float(r["mean_io_time"]) for r in rs])),
          "mean_hops":float(np.mean([float(r["mean_hops"]) for r in rs])),
          "mean_comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rs])),
        }
    return out

def aggregate_stats(reps):
    out={}
    for l in LS:
        rows=[x[str(l)] for x in reps]
        d={k:float(np.mean([r[k] for r in rows])) for k in STAT_KEYS}
        d["writes_per_query"]=d["writes"]/5000.; d["eval_writes_per_query"]=d["eval_writes"]/1000.
        d["mean_write_occupancy"]=d["write_slots"]/d["writes"] if d["writes"] else 0.
        d["mean_filler_slots_per_write"]=d["write_filler_slots"]/d["writes"] if d["writes"] else 0.
        d["final_mean_page_occupancy"]=d["final_page_slots"]/d["persisted_hubs"] if d["persisted_hubs"] else 0.
        out[str(l)]=d
    return out

def monotone(summary):
    pts=[]; best=-1e99
    for l in LS:
        r=summary[str(l)]; rec=float(r["recall_percent"])
        if rec+1e-9>=best:
            pts.append({"L":l,"recall_percent":rec,"mean_ios":float(r["mean_ios"]),"latency_us":float(r["median_latency_us"])})
            best=max(best,rec)
    return pts

def interp(summary,target):
    pts=monotone(summary)
    if target<pts[0]["recall_percent"] or target>pts[-1]["recall_percent"]: return None
    for a,b in zip(pts,pts[1:]):
        if a["recall_percent"]<=target<=b["recall_percent"]:
            x=0 if b["recall_percent"]<=a["recall_percent"]+1e-12 else (target-a["recall_percent"])/(b["recall_percent"]-a["recall_percent"])
            return {"lower":a,"upper":b,
             "mean_ios":a["mean_ios"]+x*(b["mean_ios"]-a["mean_ios"]),
             "latency_us":a["latency_us"]+x*(b["latency_us"]-a["latency_us"])}
    return pts[-1]

def matched(summary):
    out={}
    nav=summary["fill2"]
    for l in NAV_ANCHORS:
        n=nav[str(l)]; target=float(n["recall_percent"])
        item={"fill2_L":l,"recall_percent":target,"fill2_mean_ios":float(n["mean_ios"]),"fill2_latency_us":float(n["median_latency_us"]),"competitors":{}}
        for m in ("baseline","hot-30","qsev-32"):
            x=interp(summary[m],target)
            if x is None:
                item["competitors"][m]={"available":False}; continue
            item["competitors"][m]={"available":True,**x,
              "io_saving_percent":100*(1-float(n["mean_ios"])/x["mean_ios"]),
              "latency_saving_percent":100*(1-float(n["median_latency_us"])/x["latency_us"])}
        out[str(l)]=item
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ("hot-binary","qsev-binary","nav-binary","replay-queries","gt5000","index-prefix","hot-dir","qsev","ivf","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("hot_binary","qsev_binary","nav_binary","replay_queries","gt5000","index_prefix","hot_dir","qsev","ivf","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    rows,dim=fshape(args.replay_queries)
    if (rows,dim)!=(5000,DIM): raise ValueError(f"expected 5000x{DIM}, got {rows}x{dim}")
    evalq=args.work/"eval1000.fbin"; evalgt=args.work/"eval1000.gt"
    slice_fbin(args.replay_queries,evalq,4000,1000); slice_gt(args.gt5000,evalgt,4000,1000)

    hot=args.hot_dir/"hot-cache-n30.bin"
    methods=("baseline","hot-30","qsev-32","fill2")
    runs={m:[] for m in methods}; stats=[]
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(methods); order=list(methods[shift:]+methods[:shift])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                if m=="fill2":
                    rr,st=run_one(args.nav_binary,args.out,f"r{rep}-{m}",args.replay_queries,args.gt5000,args.index_prefix,ivf=args.ivf,experience=True)
                    stats.append(st)
                elif m=="qsev-32":
                    rr,_=run_one(args.qsev_binary,args.out,f"r{rep}-{m}",evalq,evalgt,args.index_prefix,qsev=args.qsev)
                elif m=="hot-30":
                    rr,_=run_one(args.hot_binary,args.out,f"r{rep}-{m}",evalq,evalgt,args.index_prefix,hot_ids=hot)
                else:
                    rr,_=run_one(args.hot_binary,args.out,f"r{rep}-{m}",evalq,evalgt,args.index_prefix)
                runs[m].append(rr); save(args.out/"runs.partial.json",runs)
    finally:
      os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in methods}
    write_stats=aggregate_stats(stats)
    ivfm=json.loads(args.ivf.with_suffix(args.ivf.suffix+".manifest.json").read_text())
    nlist=int(ivfm["nlist"])
    static_bytes=args.ivf.stat().st_size+nlist*(64+4)
    cache_ids_bytes=512*4
    nav_query_state_bytes=static_bytes+cache_ids_bytes
    qsev_bytes=args.qsev.stat().st_size
    cache_payload=30*(DIM*FLOAT_BYTES+MAX_DEGREE*4)
    result={
      "workload":"PubMed1M / MedRAG-Zipf: static training first 5K; online warm-up heldout rows 0--3999; measured heldout rows 4000--4999",
      "search":{"K":K,"Ls":list(LS),"fill2_anchor_Ls":list(NAV_ANCHORS),"beam":BEAM,"threads":THREADS,"metric":"inner_product","repetitions":args.reps,"same_graph_pq_ssd":True},
      "state_budget":{
        "fill2_static_routing_bytes":static_bytes,"fill2_cache_id_bytes":cache_ids_bytes,
        "fill2_query_state_bytes_before_container_metadata":nav_query_state_bytes,
        "persistent_hub_ids":"stored in existing graph-page slack; no graph-file growth on this layout",
        "qsev_32_bytes":qsev_bytes,"qsev_fraction_of_fill2_query_state":qsev_bytes/nav_query_state_bytes,
        "hot30_vector_plus_max_degree_edge_id_bytes":cache_payload,
        "hot30_fraction_of_fill2_query_state":cache_payload/nav_query_state_bytes,
      },
      "guardrails":[
        "All headline comparisons measure exactly the final 1,000 heldout queries.",
        "Fill2 starts cache and hub experience empty and uses only preceding heldout requests as online history.",
        "Static 16K state and hot-cache ranking use only the disjoint first 5K training queries.",
        "QSEV routing runs inside the timed query path.",
        "Cache/QSEV container overhead is excluded, which favors the competitors.",
        "All methods are order-rotated within one device-locked timing epoch."
      ],
      "summary":summary,"fill2_write_stats":write_stats,"matched_recall_against_fill2":matched(summary),
    }
    save(args.out/"frozen-paper-competitors.json",result); print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
