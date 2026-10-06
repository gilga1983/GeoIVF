#!/usr/bin/env python3
"""Distill a tiny fixed-degree RAM graph from the cached PQ-reconstruction HNSW.

The HNSW quality screen stores full reconstructed vectors and is intentionally
too large for deployment. This script keeps only level-0 neighbor IDs, producing
fixed-degree 8/16/32 adjacency files over the same 1M vertex IDs.

Format GIPQG001:
  magic[8], nvertices:u32, degree:u32, followed by nvertices*degree u32 IDs.
Missing neighbors are u32::MAX.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import faiss
import numpy as np

MAGIC=b"GIPQG001"
SENTINEL=np.uint32(0xFFFFFFFF)


def save_graph(path:Path, adj:np.ndarray):
    n,d=adj.shape
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II",n,d))
        np.asarray(adj,dtype="<u4").tofile(f)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hnsw",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--degrees",default="8,16,32")
    args=ap.parse_args()
    degrees=sorted({int(x) for x in args.degrees.split(",") if x.strip()})
    if not degrees or min(degrees)<=0:
        raise ValueError("bad degrees")

    index=faiss.read_index(str(args.hnsw))
    n=int(index.ntotal)
    h=index.hnsw
    offsets=faiss.vector_to_array(h.offsets).astype(np.int64,copy=False)
    neighbors=faiss.vector_to_array(h.neighbors).astype(np.int64,copy=False)
    level0=int(h.nb_neighbors(0))
    if len(offsets)!=n+1 or level0<max(degrees):
        raise ValueError(f"unexpected HNSW layout level0={level0}")

    args.out_dir.mkdir(parents=True,exist_ok=True)
    maxd=max(degrees)
    full=np.full((n,maxd),SENTINEL,dtype=np.uint32)
    counts=np.zeros(n,dtype=np.int32)
    dup=0
    self_edges=0

    for v in range(n):
        lo=int(offsets[v])
        hi=min(int(offsets[v+1]),lo+level0)
        seen=set()
        row=[]
        for raw in neighbors[lo:hi]:
            x=int(raw)
            if x<0:
                continue
            if x==v:
                self_edges+=1
                continue
            if x in seen:
                dup+=1
                continue
            seen.add(x)
            row.append(x)
            if len(row)==maxd:
                break
        counts[v]=len(row)
        if row:
            full[v,:len(row)]=np.asarray(row,dtype=np.uint32)

    files={}
    for d in degrees:
        p=args.out_dir/f"pq-hnsw-level0-d{d}.bin"
        save_graph(p,full[:,:d])
        files[str(d)]={"path":p.name,"bytes":p.stat().st_size}

    manifest={
        "source":str(args.hnsw),
        "vertices":n,
        "hnsw_level0_capacity":level0,
        "degrees":degrees,
        "neighbor_count":{
            "min":int(counts.min()),
            "median":float(np.median(counts)),
            "mean":float(counts.mean()),
            "p05":float(np.quantile(counts,.05)),
            "max":int(counts.max()),
        },
        "self_edges_removed":self_edges,
        "duplicate_edges_removed":dup,
        "files":files,
    }
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
