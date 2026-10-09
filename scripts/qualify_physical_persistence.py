#!/usr/bin/env python3
"""Controlled copied-index physical page-update cost for PubMed/MedCPT.

This models the write-through stage of the existing Sample1/2 controller,
with *actual* 4096B read/modify/write on a detached copy of the original
DiskANN graph file and optional fdatasync. Search uses the original Rust
controller's RAM-based persisted-exit lookup, so results are a *storage
cost prototype*, NOT validated recovery or production persistent lookup.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
import qualify_public_causal_fill2 as native

LS=(10,20,40,80,160,320)
METHODS=("core","sample2-buffered","sample2-fdatasync","sample1-fdatasync")
SOURCE_LOCK=Path.home()/".cache/geoivf/speed-device.lock"

def graph_metadata(path:Path)->dict:
    with path.open("rb") as f:
        raw=f.read(104)
    if len(raw)<104: raise ValueError("truncated graph header")
    vals=struct.unpack_from("<10Q",raw,8)
    num,dim,medoid,node_len,nps,_,_,_,file_size,_=vals
    assert num==1_000_000 and dim==768, vals
    assert nps==1 and 0 < node_len <= 4096-64, vals
    assert file_size==path.stat().st_size, (vals,path.stat().st_size)
    assert path.stat().st_size>4_000_000_000
    return {"num_points":num,"dim":dim,"node_len":node_len,"nodes_per_sector":nps,
            "file_size":file_size,"medoid":medoid,"tail_bytes":4096-node_len}

def detached_copy(prefix:Path,output:Path)->Path:
    output.parent.mkdir(parents=True,exist_ok=True)
    original=Path(str(prefix)+"_disk.index")
    clone=Path(str(output)+"_disk.index")
    subprocess.run(["cp","--reflink=auto","--sparse=always",str(original),str(clone)],check=True)
    if original.stat().st_ino==clone.stat().st_ino and original.stat().st_dev==clone.stat().st_dev:
        raise RuntimeError("graph index copy is a hardlink, refusing physical writes")
    for p in prefix.parent.glob(prefix.name+"*"):
        suffix=p.name[len(prefix.name):]
        if p==original or p.is_dir() or not suffix.startswith("_"): continue
        target=Path(str(output)+suffix)
        target.symlink_to(p.resolve())
    assert clone.is_file() and clone.stat().st_size==original.stat().st_size
    with original.open("rb") as a,clone.open("rb") as b:
        assert a.read(4096)==b.read(4096)
    return clone

def verify_record_bytes(original:Path,copy:Path,meta:dict)->None:
    with original.open("rb") as a,copy.open("rb") as b:
        assert a.read(4096)==b.read(4096)
        ids=[0,1,2,100,500,3333,10001,70001,500002,999999]
        for ident in ids:
            off=4096*(1+ident)
            a.seek(off);b.seek(off)
            if a.read(meta["node_len"])!=b.read(meta["node_len"]):
                raise AssertionError(f"graph record modified at vertex {ident}")

def parse_physical(log:Path)->dict:
    metrics={}
    for line in log.read_text(errors="replace").splitlines():
        if "PHYSICAL_PERSISTENCE_STATS " not in line:continue
        kv={}
        for token in line.split("PHYSICAL_PERSISTENCE_STATS ",1)[1].strip().split():
            if "=" in token:
                k,v=token.split("=",1)
                if v.lstrip("-").isdigit():kv[k]=int(v)
        if "L" in kv:metrics[str(kv["L"])]=kv
    if set(metrics)!=set(map(str,LS)): raise RuntimeError("missing physical stats")
    return metrics

def run(binary:Path,copy_prefix:Path,ivf:Path,
        queries:Path,groundtruth:Path,name:str,rep:int,
        output:Path,metadata:dict)->dict:
    config={
       "search_directories":[str(output)],
       "jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32",
                  "load_path":str(copy_prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(groundtruth),
          "search_list":list(LS),"beam_width":8,"recall_at":10,"num_threads":4,
          "is_flat_search":False,"distance":"inner_product",
          "vector_filters_file":None,"num_nodes_to_cache":None,
          "search_io_limit":None,"post_processor":None}
       }}]
    }
    stamp=f"r{rep}-{name}"
    inp=output/(stamp+".input.json"); out=output/(stamp+".output.json")
    log=output/(stamp+".log"); inp.write_text(json.dumps(config,indent=2)+"\n")
    env={k:v for k,v in os.environ.items() if not k.startswith("DISKANN_")}
    env.update(DISKANN_HINT_IVF_FILE=str(ivf),
               DISKANN_HINT_IVF_NPROBE="8",DISKANN_HINT_IVF_MAX_STARTS="1",
               DISKANN_EXPERIENCE_REPLAY="1", DISKANN_EXPERIENCE_WARMUP="4000",
               DISKANN_EXPERIENCE_CACHE_CAPACITY="512",
               DISKANN_EXPERIENCE_HUB_CAPACITY="0" if name=="core" else "10",
               DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR="1" if name=="sample1-fdatasync" else "2",
               DISKANN_EXPERIENCE_FILL_FROM_CACHE="0" if name=="core" else "1")
    if name!="core":
        env.update(DISKANN_EXPERIENCE_PHYSICAL_COPY=str(copy_prefix)+"_disk.index",
                   DISKANN_EXPERIENCE_PHYSICAL_NODES_PER_SECTOR=str(metadata["nodes_per_sector"]),
                   DISKANN_EXPERIENCE_PHYSICAL_NODE_LEN=str(metadata["node_len"]),
                   DISKANN_EXPERIENCE_PHYSICAL_SYNC="1" if name.endswith("fdatasync") else "0")
    t=time.monotonic()
    with log.open("w") as f:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(out)],
                       check=True,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=3600)
    rows=sorted(native.result_rows(json.loads(out.read_text())),key=lambda x:int(x["search_l"]))
    if [int(x["search_l"]) for x in rows]!=list(LS):raise RuntimeError("incomplete L grid")
    phys=parse_physical(log)
    logical=native.parse_stats(log)
    for L in LS:
        p=phys[str(L)]; ls=logical[str(L)]
        assert p["enabled"]==(name!="core")
        assert p["pages"]==ls["writes"] if name!="core" else p["pages"]==0
        assert p["eval_pages"]==ls["eval_writes"] if name!="core" else p["eval_pages"]==0
        assert p["bytes"]==p["pages"]*4096
    return {"method":name,"rep":rep,"wall_seconds":time.monotonic()-t,
            "native":rows,"logical_stats":logical,"physical_stats":phys}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","queries","groundtruth","index-prefix","ivf","work","out"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--reps",type=int,default=3)
    args=a.parse_args()
    for n in ("binary","queries","groundtruth","index_prefix","ivf","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    assert args.reps in (1,2,3)
    args.work.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    original=Path(str(args.index_prefix)+"_disk.index")
    metadata=graph_metadata(original)
    all_queries=args.work/"heldout5000.fbin"
    all_gt=args.work/"heldout5000.gt"
    native.slice_xbin(args.queries,all_queries,5000,5000,4)
    shutil.copyfile(args.groundtruth,all_gt)
    if native.xshape(all_queries)!=(5000,768):raise RuntimeError("bad measured PubMed split")
    (args.out/"original_graph_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    valid_cpus=sorted(os.sched_getaffinity(0))
    if len(valid_cpus)<4:raise RuntimeError("need four CPUs")
    state=[]
    SOURCE_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with SOURCE_LOCK.open("a") as lock:
        print("Waiting for exclusive SSD lock",flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        print("Acquired exclusive SSD lock",flush=True)
        before_hdr=original.open("rb").read(4096)
        try:
            os.sched_setaffinity(0,set(valid_cpus[:4]))
            for rep in range(args.reps):
                order=list(METHODS[rep%4:]+METHODS[:rep%4])
                for method in order:
                    prefix=args.work/f"r{rep}-{method}"/"isolated"
                    target=detached_copy(args.index_prefix,prefix)
                    result=run(args.binary,prefix,args.ivf,all_queries,all_gt,
                               method,rep,args.out,metadata)
                    verify_record_bytes(original,target,metadata)
                    shutil.rmtree(prefix.parent)
                    state.append(result)
                    (args.out/"physical-persistence.partial.json").write_text(json.dumps(state,indent=2)+"\n")
                    print(f"PHYSICAL_EPOCH_COMPLETE rep={rep} method={method}",flush=True)
        finally:
            os.sched_setaffinity(0,set(valid_cpus))
        assert original.open("rb").read(4096)==before_hdr
    (args.out/"physical-persistence.json").write_text(json.dumps({
       "protocol":"PubMed frozen causal, detached graph copies, three repetitions, four methods",
       "disclaimer":"Search reads the copied graph file; routing hint mirror is still in RAM. This isolates actual page rewrite/sync cost, not durable recovery correctness.",
       "copy_metadata":metadata,"methods":state},indent=2)+"\n")
    print("PHYSICAL_PERSISTENCE_MEASUREMENT_SUCCESS",flush=True)

if __name__=="__main__":main()
