#!/usr/bin/env python3
"""PubMed K=10 competitor RAM-scaling experiment; self-hosted SSD-only.

Exact protocol: train queries 0:5000, online warm queries 5000:9000,
measure queries 9000:10000. All variants share the same pinned DiskANN
index, K, L-grid, graph, SSD and device lock. Native/BFS and hot caches
are real RAM-resident full graph nodes. Catapult uses causal snapshots.

This is an independent scaling experiment, not a reuse of the old fake
0%-cache-hit baseline. Methods failing positive hit/read validation stop.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import struct
from pathlib import Path
import qualify_frozen_paper_competitors as hot
import qualify_final_catapult as cat

COUNTS=(30,64,128,256,512)
CAT_CAPACITIES=(80,160,320,640)
SEEDS=(0,1,2)
LS=hot.LS
assert LS==cat.LS
K=10
BEAM=8
THREADS=4
BYTES_PER_NODE=(768*4+64*4) # upper bound: vector plus maximum-degree edge IDs
CAT_HYPERPLANE_BYTES=24576
CAT_BUCKETS=256

def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n")

def cat_env(capacity:int,seed:int):
    return {"DISKANN_PAPER_CATAPULT":"1",
            "DISKANN_CATAPULT_HASHES":"8",
            "DISKANN_CATAPULT_CAPACITY":str(capacity),
            "DISKANN_CATAPULT_SEED":str(seed)}

def read_summary(rows):
    by_l={}
    for r in rows:
        key=str(int(r["search_l"]))
        by_l[key]={"recall_percent":float(r["recall"]),
                   "mean_ios":float(r["mean_ios"]),
                   "latency_us":float(r["mean_latency"]),
                   "mean_cache_hit_percent":float(r.get("cache_hit_percentage",0)),
                   "hops":float(r.get("mean_hops",0)),
                   "qps":float(r.get("qps",0))}
    return by_l

def interpolate(curve,recall,field):
    pts=sorted((v["recall_percent"],v[field]) for v in curve.values())
    for (a,x),(b,y) in zip(pts,pts[1:]):
        if a<=recall<=b:
            w=(recall-a)/(b-a) if b>a else 0
            return x+w*(y-x)
    return None

def main():
    p=argparse.ArgumentParser()
    for name in ("cache-binary","catapult-binary","nav-binary","queries","heldout-gt",
                 "training-gt","index-prefix","ivf","hot-dir","work","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reps",type=int,default=3)
    a=p.parse_args()
    for key in ("cache_binary","catapult_binary","nav_binary","queries",
                "heldout_gt","training_gt","index_prefix","ivf","hot_dir","work","out"):
        setattr(a,key,getattr(a,key).resolve())
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    if cat.fshape(a.queries)!=(10000,768): raise ValueError("unexpected query format")
    train=a.work/"train5000.fbin"
    held=a.work/"heldout5000.fbin"
    warm=a.work/"warm4000.fbin"
    evalq=a.work/"eval1000.fbin"
    warmgt=a.work/"warm4000.gt"
    evalgt=a.work/"eval1000.gt"
    cat.slice_fbin(a.queries,train,0,5000)
    cat.slice_fbin(a.queries,held,5000,5000)
    cat.slice_fbin(held,warm,0,4000)
    cat.slice_fbin(held,evalq,4000,1000)
    cat.slice_gt(a.heldout_gt,warmgt,0,4000)
    cat.slice_gt(a.heldout_gt,evalgt,4000,1000)
    for count in COUNTS:
        f=a.hot_dir/f"hot-cache-n{count}.bin"
        payload=f.read_bytes()
        if payload[:8]!=b"GIDST001" or len(payload)!=16+4*count or struct.unpack_from("<I",payload,8)[0]!=count:
            raise ValueError(f"bad cache file {f}")
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<THREADS: raise RuntimeError("need four CPU cores")
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lock_file=Path.home()/".cache/geoivf/speed-device.lock"
    lock_file.parent.mkdir(parents=True,exist_ok=True)
    methods=["baseline","core","sample2"]
    methods += [f"hot-{n}" for n in COUNTS]
    methods += [f"bfs-{n}" for n in COUNTS]
    methods += [f"cat-{n}" for n in CAT_CAPACITIES]
    results={m:[] for m in methods}
    snapshots={}
    try:
      with lock_file.open("w") as lock:
        print("Waiting for exclusive NVMe measurement lock",flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        print("Acquired exclusive NVMe measurement lock",flush=True)
        # Each Catapult budget learns its own independent causal snapshots.
        for capacity in CAT_CAPACITIES:
          for seed in SEEDS:
            env=cat_env(capacity,seed)
            path=a.work/f"cat-{capacity}-seed{seed}"
            train_snap=Path(str(path)+"-train.snapshot")
            warm_snap=Path(str(path)+"-warm.snapshot")
            cat.run(a.catapult_binary,a.out,f"train-cat-{capacity}-{seed}",train,
                a.training_gt,a.index_prefix,(10,),
                {**env,"DISKANN_CATAPULT_SNAPSHOT_DUMP":train_snap},
                skip_recall=True,threads=1)
            cat.run(a.catapult_binary,a.out,f"warm-cat-{capacity}-{seed}",warm,
                warmgt,a.index_prefix,(10,),
                {**env,"DISKANN_CATAPULT_SNAPSHOT_LOAD":train_snap,
                 "DISKANN_CATAPULT_SNAPSHOT_DUMP":warm_snap},
                skip_recall=True,threads=1)
            entries=cat.snapshot_entries(warm_snap)
            if entries<=0: raise RuntimeError(f"Catapult {capacity}/{seed}: empty snapshot")
            snapshots[(capacity,seed)]=warm_snap
        save(a.out/"snapshot_manifest.json",{
            f"{cap}-{seed}":{"entries":cat.snapshot_entries(f),
                            "bytes":f.stat().st_size}
            for (cap,seed),f in snapshots.items()})
        for rep in range(a.reps):
          seed=SEEDS[rep%len(SEEDS)]
          order=methods[rep%len(methods):]+methods[:rep%len(methods)]
          print(f"rep={rep} seed={seed} method_order={order}",flush=True)
          for method in order:
            tag=f"r{rep}-{method}"
            if method=="baseline":
                hot.run_one(a.cache_binary,a.out,tag+"-warm",warm,warmgt,
                            a.index_prefix)
                rows,_=hot.run_one(a.cache_binary,a.out,tag,evalq,evalgt,
                                   a.index_prefix)
            elif method.startswith("hot-"):
                count=int(method.split("-")[1])
                cache_path=a.hot_dir/f"hot-cache-n{count}.bin"
                hot.run_one(a.cache_binary,a.out,tag+"-warm",warm,warmgt,
                            a.index_prefix,hot_ids=cache_path)
                rows,_=hot.run_one(a.cache_binary,a.out,tag,evalq,evalgt,
                                   a.index_prefix,hot_ids=cache_path)
            elif method.startswith("bfs-"):
                count=int(method.split("-")[1])
                hot.run_one(a.cache_binary,a.out,tag+"-warm",warm,warmgt,
                            a.index_prefix,cache_nodes=count)
                rows,_=hot.run_one(a.cache_binary,a.out,tag,evalq,evalgt,
                                   a.index_prefix,cache_nodes=count)
            elif method.startswith("cat-"):
                capacity=int(method.split("-")[1])
                env={**cat_env(capacity,seed),
                     "DISKANN_CATAPULT_SNAPSHOT_LOAD":snapshots[(capacity,seed)]}
                cat.run(a.catapult_binary,a.out,tag+"-storagewarm",warm,warmgt,
                        a.index_prefix,LS,
                        {**env,"DISKANN_CATAPULT_FREEZE":"1"},skip_recall=True)
                rows,_=cat.run(a.catapult_binary,a.out,tag,evalq,evalgt,
                               a.index_prefix,LS,env)
            else:
                env={"DISKANN_HINT_IVF_FILE":a.ivf,
                     "DISKANN_HINT_IVF_NPROBE":"8",
                     "DISKANN_HINT_IVF_MAX_STARTS":"1",
                     "DISKANN_EXPERIENCE_REPLAY":"1",
                     "DISKANN_EXPERIENCE_WARMUP":"4000",
                     "DISKANN_EXPERIENCE_CACHE_CAPACITY":"512",
                     "DISKANN_EXPERIENCE_HUB_CAPACITY":"0" if method=="core" else "10",
                     "DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR":"2",
                     "DISKANN_EXPERIENCE_FILL_FROM_CACHE":"1"}
                rows,_=cat.run(a.nav_binary,a.out,tag,held,a.heldout_gt,
                               a.index_prefix,LS,env,experience=True)
            summary=read_summary(rows)
            if len(summary)!=len(LS): raise RuntimeError(f"{tag}: incomplete L grid")
            results[method].append(summary)
            save(a.out/"runs.partial.json",results)
        anchor=results["sample2"]
        if len(anchor)!=a.reps: raise ValueError("incomplete Sample2 reference")
        comparisons={}
        for method,curves in results.items():
            anchors=[]
            for rep,reference in enumerate(anchor):
                ref=reference
                cand=curves[rep]
                for L in (10,20,40,80,160,320):
                    rec=ref[str(L)]["recall_percent"]
                    other=interpolate(cand,rec,"mean_ios")
                    if other is None: continue
                    anchors.append({"rep":rep,"anchor_L":L,"recall":rec,
                      "reference_reads":ref[str(L)]["mean_ios"],
                      "comparison_reads":other,
                      "sample2_saving_percent":100*(1-ref[str(L)]["mean_ios"]/other)})
            if not anchors: raise RuntimeError(f"no matched anchors for {method}")
            comparisons[method]={
                "matched_anchors":anchors,
                "mean_sample2_io_saving_percent":sum(
                    x["sample2_saving_percent"] for x in anchors)/len(anchors),
                "measured_repetitions":len(curves),
            }
        memory={"core_query_payload_bytes":102940,
                "hot_node_memory_upper_estimate_bytes":{str(n):n*BYTES_PER_NODE for n in COUNTS},
                "catapult_reserved_bytes":{
                    str(cap):CAT_BUCKETS*cap*4+CAT_HYPERPLANE_BYTES
                    for cap in CAT_CAPACITIES}}
        save(a.out/"competitor-memory-sweep.json",{
           "protocol":{"static_train":5000,"online_warmup":4000,"measured":1000,
              "K":K,"beam":BEAM,"threads":THREADS,"Ls":list(LS),
              "repetitions":a.reps,"disk_lock":str(lock_file),
              "order_rotated":True,"fixed_graph_and_ssd":True},
           "notes":["Hot and BFS caches must record positive actual hit rates",
                    "Catapult trains independent causal snapshots at each capacity",
                    "All improvements reported at matched recall, using interpolated reads",
                    "RAM comparison is auxiliary payload, excludes containers and allocator overhead"],
           "memory":memory,"snapshots":{
               f"{c}-{s}":cat.snapshot_entries(f) for (c,s),f in snapshots.items()},
           "results":results,"matched_comparisons":comparisons})
        print("COMPETITOR_MEMORY_SWEEP_SUCCESS",flush=True)
    finally:
      os.sched_setaffinity(0,set(allowed))

if __name__=="__main__":
    main()
