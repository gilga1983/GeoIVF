#!/usr/bin/env python3
"""Build paired 16K-winner payloads for start-selected vs traversed-central placement.

For each warm query:
  winner = actual rank-1 result from the exact 16K + support-2 L160 search.

Two placement policies see the exact same winner stream:
  start:
    the first expanded vertex in the exact trace (the deployed 16K start).
  traversed:
    the last expanded vertex in the same trace that belongs to the canonical
    16K landmark vocabulary. If none appears later, this naturally falls back
    to the start landmark.

Each selected hub keeps the first ten distinct winners chronologically.
"""
from __future__ import annotations

import argparse,json,struct
from pathlib import Path
import numpy as np

MAGIC=b"GIVTX001"
IVF_MAGIC=b"GHIVF001"
TAGS=np.asarray([0xFFFFFFFA,0xFFFFFFF9,0xFFFFFFF8],dtype=np.uint32)

def load_payload(path):
    raw=path.read_bytes()
    if len(raw)<24 or raw[:8]!=MAGIC:raise ValueError("bad payload")
    n,v,s,m=struct.unpack_from("<IIII",raw,8);off=24
    vids=np.frombuffer(raw,dtype="<u4",count=v,offset=off).copy();off+=v*4
    count=n*v*s
    if len(raw)!=off+count*4:raise ValueError("payload size mismatch")
    data=np.frombuffer(raw,dtype="<u4",count=count,offset=off).reshape(n,v,s).copy()
    return int(n),int(v),int(s),int(m),vids,data

def load_ivf_ids(path):
    raw=path.read_bytes()
    if len(raw)<24 or raw[:8]!=IVF_MAGIC:raise ValueError("bad IVF")
    nlist,nchild,total,res=struct.unpack_from("<IIII",raw,8)
    if res!=0 or total!=nlist+nchild:raise ValueError("bad IVF shape")
    off=24
    med=np.frombuffer(raw,dtype="<u4",count=nlist,offset=off).copy();off+=4*nlist
    off+=4*(nlist+1)
    child=np.frombuffer(raw,dtype="<u4",count=nchild,offset=off).copy()
    ids=np.concatenate([med,child]).astype(np.uint32,copy=False)
    if len(np.unique(ids))!=total:raise ValueError("duplicate IVF landmark")
    return set(map(int,ids)),int(total)

def load_results(path):
    with path.open("rb") as f:
        r,k=struct.unpack("<II",f.read(8));a=np.fromfile(f,dtype="<u4",count=r*k)
    if a.size!=r*k:raise ValueError("truncated results")
    return a.reshape(r,k)

def load_trace(path,expected):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=expected:raise ValueError(f"trace rows {len(rows)} != {expected}")
    for i,r in enumerate(rows):
        if int(r["query"])!=i or not r["ids"]:raise ValueError("bad trace row")
    return rows

def place(trace,landmarks,mode):
    ids=[int(x) for x in trace["ids"]]
    if mode=="start":
        return ids[0]
    if mode=="traversed":
        cand=[x for x in ids if x in landmarks]
        return cand[-1] if cand else ids[0]
    raise ValueError(mode)

def learn(n,rows,results,landmarks,mode,warm_rows):
    buckets={}
    duplicates=0
    same_as_start=0
    later_pos=[]
    for i in range(warm_rows):
        hub=place(rows[i],landmarks,mode)
        start=int(rows[i]["ids"][0])
        if hub==start:same_as_start+=1
        if mode=="traversed":
            ids=[int(x) for x in rows[i]["ids"]]
            later_pos.append(max(j for j,x in enumerate(ids) if x==hub))
        winner=int(results[i,0])
        if not (0<=hub<n and 0<=winner<n):raise ValueError("ID outside graph")
        b=buckets.setdefault(hub,[])
        if winner in b:
            duplicates+=1;continue
        if len(b)<10:b.append(winner)
    tail=np.full((n,12),np.uint32(0xFFFFFFFF),dtype=np.uint32)
    for h,b in buckets.items():tail[h,:len(b)]=np.asarray(b,dtype=np.uint32)
    counts=np.asarray([len(v) for v in buckets.values()],dtype=np.int64)
    return tail.reshape(n,3,4),{
      "active_hubs":len(buckets),
      "duplicate_contributions_ignored":duplicates,
      "same_as_start_fraction":same_as_start/warm_rows,
      "traversed_landmark_position":{
        "mean":float(np.mean(later_pos)) if later_pos else 0.0,
        "median":float(np.median(later_pos)) if later_pos else 0.0,
        "p95":float(np.quantile(later_pos,.95)) if later_pos else 0.0,
      },
      "winner_count_per_active_hub":{
        "min":int(counts.min()) if len(counts) else 0,
        "median":float(np.median(counts)) if len(counts) else 0.0,
        "mean":float(counts.mean()) if len(counts) else 0.0,
        "p95":float(np.quantile(counts,.95)) if len(counts) else 0.0,
        "max":int(counts.max()) if len(counts) else 0,
      }
    }

def write(path,n,v,s,m,vids,base,tail):
    combined=np.concatenate([base,tail],axis=1)
    out_ids=np.concatenate([vids,TAGS])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC);f.write(struct.pack("<IIII",n,v+3,s,m))
        out_ids.astype("<u4",copy=False).tofile(f)
        combined.astype("<u4",copy=False).tofile(f)

def main():
    ap=argparse.ArgumentParser()
    for n in ("base-payload","ivf","trace","results","out-dir"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--warm-rows",type=int,default=4000)
    args=ap.parse_args()
    n,v,s,m,vids,base=load_payload(args.base_payload.resolve())
    if v!=5 or s!=4:raise ValueError("expected 5x4 base")
    results=load_results(args.results.resolve())
    rows=load_trace(args.trace.resolve(),results.shape[0])
    landmarks,total=load_ivf_ids(args.ivf.resolve())
    if args.warm_rows>results.shape[0]:raise ValueError("warm rows")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={"landmark_vocabulary":total,"warm_rows":args.warm_rows,"placements":{}}
    for mode in ("start","traversed"):
        tail,meta=learn(n,rows,results,landmarks,mode,args.warm_rows)
        p=args.out_dir/f"{mode}.bin";write(p,n,v,s,m,vids,base,tail)
        meta["file"]=p.name
        manifest["placements"][mode]=meta
        print(json.dumps({"mode":mode,**meta}),flush=True)
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=="__main__":main()
