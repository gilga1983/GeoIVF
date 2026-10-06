#!/usr/bin/env python3
"""Generate DiskANN start-point files by walking a compact graph in PQ space.

All walks start from the exact deployed NavHint, taken from the first expansion
of held-out trace-enabled canonical searches. The walk is entirely RAM-side:
adjacency IDs plus DiskANN's resident PQ codes/pivots.

Search policy: best-first over PQ inner-product score. Expand at most B vertices,
score unseen fixed-degree neighbors, then return the best S discovered IDs.
"""
from __future__ import annotations

import argparse
import heapq
import json
import struct
import time
from pathlib import Path

import numpy as np

from analyze_pq_semantic_filter import fbin_memmap, load_pq, lut_for_query, pq_scores

MAGIC=b"GIPQG001"
SENTINEL=0xFFFFFFFF


def load_graph(path:Path):
    raw=np.memmap(path,mode="r",dtype="u1")
    if len(raw)<16 or bytes(raw[:8])!=MAGIC:
        raise ValueError("bad compact graph")
    n,d=struct.unpack("<II",bytes(raw[8:16]))
    expected=16+n*d*4
    if len(raw)!=expected:
        raise ValueError("compact graph size mismatch")
    adj=np.memmap(path,mode="r",dtype="<u4",offset=16,shape=(n,d))
    return n,d,adj


def write_starts(path:Path,rows):
    a=np.asarray(rows,dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II",a.shape[0],a.shape[1]))
        a.tofile(f)


def trace_starts(path:Path,nq:int):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=nq:
        raise ValueError("trace row mismatch")
    out=[]
    for qi,r in enumerate(rows):
        if int(r["query"])!=qi or not r["ids"]:
            raise ValueError("invalid trace")
        out.append(int(r["ids"][0]))
    return np.asarray(out,dtype=np.uint32)


def walk(q,lut,codes,adj,start,budget,seed_count):
    # Heap uses negative similarity so best score pops first.
    s0=float(pq_scores([start],lut,codes)[0])
    scores={int(start):s0}
    heap=[(-s0,int(start))]
    expanded=set()
    comparisons=1

    while heap and len(expanded)<budget:
        neg,v=heapq.heappop(heap)
        if v in expanded:
            continue
        expanded.add(v)
        nbrs=[]
        for raw in adj[v]:
            x=int(raw)
            if x==SENTINEL or x in scores:
                continue
            nbrs.append(x)
        if not nbrs:
            continue
        vals=pq_scores(nbrs,lut,codes)
        comparisons+=len(nbrs)
        for x,score in zip(nbrs,vals):
            sf=float(score)
            scores[x]=sf
            heapq.heappush(heap,(-sf,x))

    ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0]))
    ids=[x for x,_ in ranked[:seed_count]]
    # We always have at least the original start.
    if len(ids)<seed_count:
        # Start-point file requires fixed width and no duplicates. Fill by
        # additional ranked discoveries if possible; fail rather than invent.
        raise RuntimeError("RAM walk discovered too few unique seeds")
    return ids,comparisons,len(expanded),len(scores)


def main():
    ap=argparse.ArgumentParser()
    for name in ("queries","pq-pivots","pq-codes","graph","trace","out-dir"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--query-offset",type=int,default=9000)
    ap.add_argument("--query-count",type=int,default=1000)
    ap.add_argument("--budgets",default="16,32,64,128")
    ap.add_argument("--seed-counts",default="1,2,4")
    args=ap.parse_args()

    budgets=sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    seed_counts=sorted({int(x) for x in args.seed_counts.split(",") if x.strip()})
    if min(budgets)<=0 or min(seed_counts)<=0:
        raise ValueError("invalid sweep")

    qall=fbin_memmap(args.queries.resolve())
    queries=np.asarray(qall[args.query_offset:args.query_offset+args.query_count],dtype=np.float32)
    pivots,offsets,codes=load_pq(args.pq_pivots.resolve(),args.pq_codes.resolve())
    n,d,adj=load_graph(args.graph.resolve())
    if n!=codes.shape[0] or len(queries)!=args.query_count:
        raise ValueError("shape mismatch")

    starts=trace_starts(args.trace.resolve(),len(queries))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_starts(args.out_dir/"navhint-seed1.bin",starts[:,None])

    configs={}
    for budget in budgets:
        # Do one walk to max requested seed width, then prefix the ranked seeds.
        maxs=max(seed_counts)
        rows=np.empty((len(queries),maxs),dtype=np.uint32)
        cmp_total=exp_total=disc_total=0
        t0=time.perf_counter()
        for qi,q in enumerate(queries):
            lut=lut_for_query(q,pivots,offsets)
            ids,cmps,expanded,discovered=walk(
                q,lut,codes,adj,int(starts[qi]),budget,maxs
            )
            rows[qi]=ids
            cmp_total+=cmps
            exp_total+=expanded
            disc_total+=discovered
        elapsed=time.perf_counter()-t0
        for sc in seed_counts:
            p=args.out_dir/f"ramwalk-b{budget}-s{sc}.bin"
            write_starts(p,rows[:,:sc])
            configs[f"b{budget}_s{sc}"]={
                "file":p.name,
                "budget":budget,
                "seed_count":sc,
                "walk_seconds_total":elapsed,
                "walk_us_per_query":1e6*elapsed/len(queries),
                "mean_pq_comparisons":cmp_total/len(queries),
                "mean_expanded_ram_vertices":exp_total/len(queries),
                "mean_discovered_ram_vertices":disc_total/len(queries),
            }
        print(json.dumps({"budget":budget,"walk_us_per_query":1e6*elapsed/len(queries),"mean_comparisons":cmp_total/len(queries)}),flush=True)

    manifest={
        "graph":str(args.graph),
        "vertices":n,
        "degree":d,
        "queries":len(queries),
        "query_offset":args.query_offset,
        "baseline_file":"navhint-seed1.bin",
        "configs":configs,
        "policy":"best-first PQ-inner-product walk from exact deployed NavHint; fixed expansion budget",
    }
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
