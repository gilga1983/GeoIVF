#!/usr/bin/env python3
"""Final paper comparison: CatapultDB vs frozen NavHints controller.

Protocol:
* static training history: first 5K MedRAG-Zipf requests;
* online warm-up history: first 4K requests of the disjoint heldout 5K;
* measured suffix: final 1K heldout requests;
* Catapult warm-up is materialized into a snapshot before timing;
* NavHints replays all heldout 5K causally and its benchmark reports only the
  final 1K after DISKANN_EXPERIENCE_WARMUP=4000;
* baseline/entry-only run directly on exactly the same final 1K.
"""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,12,20,22,24,40,44,80,88,160,176,320)
ANCHORS=(10,20,40,80,160,320)
SEEDS=(0,1,2)
DIM=768
STAT_KEYS=(
 "cache_capacity","hub_capacity","sample_denominator","fill","cache_inserts","cache_skips",
 "cache_evictions","active_direct_hubs","persisted_hubs","direct_learned",
 "direct_duplicates","direct_full","sample_trials","sample_accepts","sample_rejects",
 "fifo_evictions","writes","eval_writes","write_slots","write_direct_slots",
 "write_filler_slots","eval_write_slots","eval_write_filler_slots","final_page_slots",
)

def save(p,o):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(o,indent=2)+"\n")

def fshape(p):
    p=Path(p)
    with p.open("rb") as f: raw=f.read(8)
    if len(raw)!=8: raise ValueError(f"bad fbin header {p}")
    rows,dim=struct.unpack("<II",raw)
    if p.stat().st_size!=8+rows*dim*4: raise ValueError(f"bad fbin size {p}")
    return rows,dim

def slice_fbin(src,dst,start,count):
    src,dst=Path(src),Path(dst)
    rows,dim=fshape(src)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad fbin slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4); fo.write(struct.pack("<II",count,dim))
        rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)

def slice_gt(src,dst,start,count):
    src,dst=Path(src),Path(dst)
    with src.open("rb") as f: raw=f.read(8)
    if len(raw)!=8: raise ValueError("bad gt header")
    rows,k=struct.unpack("<II",raw)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad gt slice")
    rb=k*4; ids_bytes=rows*rb; size=src.stat().st_size
    ids_only=size==8+ids_bytes
    ids_dists=size==8+2*ids_bytes
    if not ids_only and not ids_dists:
        raise ValueError(f"unexpected gt size {size} for {(rows,k)}")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        fi.seek(8+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated gt ids")
            fo.write(b); rem-=len(b)
        if ids_dists:
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
    for line in Path(log).read_text().splitlines():
        if "EXPERIENCE_STATS " not in line: continue
        kv={}
        for tok in line.split("EXPERIENCE_STATS ",1)[1].split():
            if "=" in tok:
                k,v=tok.split("=",1); kv[k]=int(v)
        if "L" in kv: out[str(kv["L"])]={k:kv[k] for k in STAT_KEYS if k in kv}
    if set(out)!=set(map(str,LS)): raise ValueError(f"missing experience stats {sorted(out)}")
    return out

def cfg(out,queries,gt,prefix,ls,threads=THREADS):
    return {"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(ls),
       "beam_width":BEAM,"recall_at":K,"num_threads":threads,"is_flat_search":False,
       "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
       "search_io_limit":None,"post_processor":None}}}]}

def run(binary,out,tag,queries,gt,prefix,ls,extra,*,skip_recall=False,threads=THREADS,experience=False):
    inp=out/f"{tag}.input.json"; op=out/f"{tag}.output.json"; log=out/f"{tag}.log"
    save(inp,cfg(out,queries,gt,prefix,ls,threads))
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"): env.pop(n,None)
    if skip_recall: env["DISKANN_SKIP_RECALL"]="1"
    env.update({k:str(v) for k,v in extra.items()})
    t=time.monotonic()
    with log.open("w") as lf:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
                       stdout=lf,stderr=subprocess.STDOUT,env=env,check=True,timeout=2400)
    print(f"completed {tag} in {time.monotonic()-t:.1f}s",flush=True)
    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda x:int(x["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(ls): raise ValueError(f"{tag}: bad L grid")
    return rr,(parse_stats(log) if experience else None)

def snapshot_entries(path):
    raw=Path(path).read_bytes()
    if len(raw)<32 or raw[:8]!=b"GICAT001": raise ValueError("bad catapult snapshot")
    return struct.unpack("<I",raw[28:32])[0]

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "rounds":len(rs),
          "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
          "qps":float(np.median([float(r["qps"]) for r in rs])),
          "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rs])),
          "hops":float(np.mean([float(r["mean_hops"]) for r in rs])),
          "catapult_usage_percent":float(np.mean([float(r.get("catapult_usage_percentage",0)) for r in rs])),
          "catapult_starts":float(np.mean([float(r.get("mean_catapult_starts",0)) for r in rs])),
        }
    return out

def aggregate_stats(reps):
    out={}
    for l in LS:
        rows=[x[str(l)] for x in reps]
        d={k:float(np.mean([r[k] for r in rows])) for k in STAT_KEYS}
        d["writes_per_query"]=d["writes"]/5000.
        d["eval_writes_per_query"]=d["eval_writes"]/1000.
        d["acceptance_fraction"]=d["sample_accepts"]/d["sample_trials"] if d["sample_trials"] else 0.
        d["final_mean_page_occupancy"]=d["final_page_slots"]/d["persisted_hubs"] if d["persisted_hubs"] else 0.
        out[str(l)]=d
    return out

def monotone(s):
    pts=[]; best=-1e99
    for l in LS:
        r=s[str(l)]; rec=float(r["recall_percent"])
        if rec+1e-9>=best:
            pts.append((rec,r)); best=max(best,rec)
    return pts

def interp(s,target,field):
    pts=monotone(s)
    if not pts or target<pts[0][0] or target>pts[-1][0]: return None
    for (a,ra),(b,rb) in zip(pts,pts[1:]):
        if a<=target<=b:
            x=0 if b<=a+1e-12 else (target-a)/(b-a)
            return float(ra[field])+x*(float(rb[field])-float(ra[field]))
    return float(pts[-1][1][field])

def matched(summary,anchor):
    out={}
    for l in ANCHORS:
        row=summary[anchor][str(l)]; target=float(row["recall_percent"])
        item={"anchor_L":l,"recall_percent":target,"anchor":anchor,"comparisons":{}}
        for m in summary:
            io=interp(summary[m],target,"mean_ios"); la=interp(summary[m],target,"latency_us")
            if io is None or la is None:
                item["comparisons"][m]={"available":False}; continue
            item["comparisons"][m]={
              "available":True,"mean_ios":io,"latency_us":la,
              "anchor_io_saving_percent":100*(1-float(row["mean_ios"])/io),
              "anchor_latency_saving_percent":100*(1-float(row["latency_us"])/la),
            }
        out[str(l)]=item
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ("catapult-binary","nav-binary","queries","heldout-gt","training-gt",
              "index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("catapult_binary","nav_binary","queries","heldout_gt","training_gt",
              "index_prefix","ivf_16k","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if fshape(a.queries)!=(10000,DIM): raise ValueError("unexpected workload")

    train=a.work/"train5000.fbin"; held=a.work/"heldout5000.fbin"
    warm=a.work/"warm4000.fbin"; evalq=a.work/"eval1000.fbin"
    warmgt=a.work/"warm4000.gt"; evalgt=a.work/"eval1000.gt"
    slice_fbin(a.queries,train,0,5000); slice_fbin(a.queries,held,5000,5000)
    slice_fbin(held,warm,0,4000); slice_fbin(held,evalq,4000,1000)
    slice_gt(a.heldout_gt,warmgt,0,4000); slice_gt(a.heldout_gt,evalgt,4000,1000)

    # Catapult receives the same online warm-up horizon before final timing.
    warmed={}
    snapmeta=[]
    for seed in SEEDS:
        train_snap=a.work/f"catapult-s{seed}-train.snapshot"
        warm_snap=a.work/f"catapult-s{seed}-warm.snapshot"
        base_env={"DISKANN_PAPER_CATAPULT":"1","DISKANN_CATAPULT_HASHES":"8",
                  "DISKANN_CATAPULT_CAPACITY":"40","DISKANN_CATAPULT_SEED":seed}
        run(a.catapult_binary,a.out,f"train-cat-s{seed}",train,a.training_gt,a.index_prefix,(10,),
            {**base_env,"DISKANN_CATAPULT_SNAPSHOT_DUMP":train_snap},
            skip_recall=True,threads=1)
        run(a.catapult_binary,a.out,f"warm-cat-s{seed}",warm,warmgt,a.index_prefix,(10,),
            {**base_env,"DISKANN_CATAPULT_SNAPSHOT_LOAD":train_snap,
             "DISKANN_CATAPULT_SNAPSHOT_DUMP":warm_snap},
            skip_recall=True,threads=1)
        warmed[seed]=warm_snap
        snapmeta.append({"seed":seed,"train_entries":snapshot_entries(train_snap),
                         "warm_entries":snapshot_entries(warm_snap),
                         "warm_snapshot_bytes":warm_snap.stat().st_size})

    methods=("baseline","ivf","core","sample2","catapult")
    runs={m:[] for m in methods}; stats=[]
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(a.reps):
            seed=SEEDS[rep%len(SEEDS)]
            order=list(methods[rep%len(methods):]+methods[:rep%len(methods)])
            print(f"rep={rep} seed={seed} order={' '.join(order)}",flush=True)
            for m in order:
                if m=="catapult":
                    rr,_=run(a.catapult_binary,a.out,f"r{rep}-catapult",evalq,evalgt,a.index_prefix,LS,{
                      "DISKANN_PAPER_CATAPULT":"1","DISKANN_CATAPULT_HASHES":"8",
                      "DISKANN_CATAPULT_CAPACITY":"40","DISKANN_CATAPULT_SEED":seed,
                      "DISKANN_CATAPULT_SNAPSHOT_LOAD":warmed[seed],
                    })
                elif m=="baseline":
                    rr,_=run(a.nav_binary,a.out,f"r{rep}-baseline",evalq,evalgt,a.index_prefix,LS,{})
                elif m=="ivf":
                    rr,_=run(a.nav_binary,a.out,f"r{rep}-ivf",evalq,evalgt,a.index_prefix,LS,{
                      "DISKANN_HINT_IVF_FILE":a.ivf_16k,"DISKANN_HINT_IVF_NPROBE":"8",
                      "DISKANN_HINT_IVF_MAX_STARTS":"1",
                    })
                else:
                    hub=0 if m=="core" else 10
                    rr,st=run(a.nav_binary,a.out,f"r{rep}-{m}",held,a.heldout_gt,a.index_prefix,LS,{
                      "DISKANN_HINT_IVF_FILE":a.ivf_16k,"DISKANN_HINT_IVF_NPROBE":"8",
                      "DISKANN_HINT_IVF_MAX_STARTS":"1","DISKANN_EXPERIENCE_REPLAY":"1",
                      "DISKANN_EXPERIENCE_WARMUP":"4000","DISKANN_EXPERIENCE_CACHE_CAPACITY":"512",
                      "DISKANN_EXPERIENCE_HUB_CAPACITY":str(hub),
                      "DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR":"2",
                      "DISKANN_EXPERIENCE_FILL_FROM_CACHE":"1",
                    },experience=True)
                    if m=="sample2": stats.append(st)
                runs[m].append(rr); save(a.out/"runs.partial.json",runs)
    finally:
      os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in methods}
    ivfm=json.loads(a.ivf_16k.with_suffix(a.ivf_16k.suffix+".manifest.json").read_text())
    nav_static=a.ivf_16k.stat().st_size+int(ivfm["nlist"])*(64+4)
    result={
      "protocol":{"static_train":5000,"online_warmup":4000,"measured":1000,
                  "K":K,"Ls":list(LS),"beam":BEAM,"threads":THREADS,
                  "same_graph_pq_ssd":True,"order_rotated":True},
      "navhints":{"controller_blob":"2cd0b5a2cbd8ea059cdd1dc0745a9679d2f6f6eb",
                  "static_routing_bytes":nav_static,"cache_id_bytes":2048,
                  "core":"16K entry + 512 recent-result IDs",
                  "full":"core + Sample2 10-ID hub FIFO in page slack"},
      "catapult":{"hashes":8,"bucket_capacity":40,"seeds":list(SEEDS),
                  "snapshots":snapmeta,
                  "accounting_note":"snapshot/hyperplane payload reported separately; container overhead omitted, favoring Catapult"},
      "summary":summary,
      "sample2_write_stats":aggregate_stats(stats),
      "matched_at_core":matched(summary,"core"),
      "matched_at_sample2":matched(summary,"sample2"),
    }
    save(a.out/"final-catapult.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
