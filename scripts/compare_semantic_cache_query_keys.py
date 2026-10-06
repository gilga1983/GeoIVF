#!/usr/bin/env python3
"""Compare fp16 versus DiskANN-PQ representations for semantic-cache lookup.

For each evaluated query, scan the same rolling cache using:
  1) fp16 stored cached query vectors (reference)
  2) one 64-byte DiskANN PQ code per cached query

The current query remains full precision. PQ lookup uses the same per-query LUT
formula as DiskANN and scores each cached query's PQ reconstruction.

For each representation, emit NavHint + oracle previous-result IDs so downstream
DiskANN can measure how much routing value survives query-key compression.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from analyze_pq_semantic_filter import fbin_memmap, gt_ids, load_pq, lut_for_query

WARM_Q0 = 5000
EVAL_Q0 = 9000
EVAL_N = 1000


def trace_starts(path: Path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=EVAL_N:
        raise ValueError(f"expected {EVAL_N} trace rows")
    out=np.empty(EVAL_N,dtype=np.uint32)
    for i,r in enumerate(rows):
        if int(r["query"])!=i or not r["ids"]:
            raise ValueError("bad trace")
        out[i]=int(r["ids"][0])
    return out


def write_starts(path: Path, rows: np.ndarray):
    a=np.asarray(rows,dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II",a.shape[0],a.shape[1]))
        a.tofile(f)


def encode_queries_pq(q: np.ndarray, pivots: np.ndarray, offsets: np.ndarray, block=512):
    """Encode queries by nearest L2 codeword independently in each PQ chunk."""
    n=q.shape[0]
    chunks=len(offsets)-1
    codes=np.empty((n,chunks),dtype=np.uint8)
    for j in range(chunks):
        a,b=int(offsets[j]),int(offsets[j+1])
        cent=pivots[:,a:b].astype(np.float32,copy=False)
        cent_norm=np.einsum("ij,ij->i",cent,cent,optimize=True)
        for s in range(0,n,block):
            e=min(n,s+block)
            x=np.asarray(q[s:e,a:b],dtype=np.float32)
            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c. The x norm is common.
            scores=x @ cent.T
            dist=cent_norm[None,:] - 2.0*scores
            codes[s:e,j]=np.argmin(dist,axis=1).astype(np.uint8)
    return codes


def score_pq_cache(q, codes, pivots, offsets):
    lut=lut_for_query(q,pivots,offsets)
    out=np.zeros(codes.shape[0],dtype=np.float32)
    rows=np.arange(codes.shape[0])
    for j in range(codes.shape[1]):
        out += lut[j,codes[:,j]]
    return out


def starts_from_match(nav, gt, matches_abs, result_count):
    rows=np.empty((EVAL_N,1+result_count),dtype=np.uint32)
    rows[:,0]=nav
    for ei,absq in enumerate(matches_abs):
        gtrow=int(absq)-WARM_Q0
        seen={int(nav[ei])}
        chosen=[]
        for raw in gt[gtrow]:
            rid=int(raw)
            if rid in seen:
                continue
            seen.add(rid)
            chosen.append(rid)
            if len(chosen)==result_count:
                break
        if len(chosen)!=result_count:
            raise RuntimeError("insufficient distinct results")
        rows[ei,1:]=np.asarray(chosen,dtype=np.uint32)
    return rows


def main():
    ap=argparse.ArgumentParser()
    for name in ("queries","gt5000","pq-pivots","pq-codes","trace","out-dir"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--capacities",default="128,512,2048")
    ap.add_argument("--result-count",type=int,default=10)
    args=ap.parse_args()

    caps=sorted({int(x) for x in args.capacities.split(",") if x.strip()})
    queries=fbin_memmap(args.queries.resolve())
    gt=gt_ids(args.gt5000.resolve())
    pivots,offsets,_=load_pq(args.pq_pivots.resolve(),args.pq_codes.resolve())
    if queries.shape!=(10000,768) or gt.shape[0]!=5000:
        raise ValueError("unexpected workload shape")

    nav=trace_starts(args.trace.resolve())
    qcache32=np.asarray(queries[WARM_Q0:WARM_Q0+gt.shape[0]],dtype=np.float32)
    qcache16=qcache32.astype(np.float16)
    qeval=np.asarray(queries[EVAL_Q0:EVAL_Q0+EVAL_N],dtype=np.float32)

    print("encoding cached queries with DiskANN PQ codebook",flush=True)
    tenc=time.perf_counter()
    qcodes=encode_queries_pq(qcache32,pivots,offsets)
    enc_s=time.perf_counter()-tenc
    if qcodes.shape[1]!=64:
        print(f"warning: PQ chunks={qcodes.shape[1]}",flush=True)

    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_starts(args.out_dir/"navhint-only.bin",nav[:,None])

    result={
        "policy":"rolling semantic cache; compare fp16 query key versus DiskANN PQ-coded query key",
        "pq":{
            "chunks":int(qcodes.shape[1]),
            "bytes_per_cached_query":int(qcodes.shape[1]),
            "encode_seconds_for_5000_queries":enc_s,
        },
        "capacities":{},
    }

    for cap in caps:
        fp_match=np.empty(EVAL_N,dtype=np.int32)
        pq_match=np.empty(EVAL_N,dtype=np.int32)
        fp_best=np.empty(EVAL_N,dtype=np.float32)
        pq_chosen_true=np.empty(EVAL_N,dtype=np.float32)
        fp_scan=0.0
        pq_scan=0.0

        for ei,q in enumerate(qeval):
            absq=EVAL_Q0+ei
            rel=absq-WARM_Q0
            lo=max(0,rel-cap); hi=rel

            t=time.perf_counter()
            fp_cache=np.asarray(qcache16[lo:hi],dtype=np.float32)
            sims=fp_cache @ q
            li=int(np.argmax(sims))
            fp_scan += time.perf_counter()-t
            fp_match[ei]=WARM_Q0+lo+li
            fp_best[ei]=float(sims[li])

            t=time.perf_counter()
            ps=score_pq_cache(q,qcodes[lo:hi],pivots,offsets)
            pli=int(np.argmax(ps))
            pq_scan += time.perf_counter()-t
            pq_match[ei]=WARM_Q0+lo+pli
            # Evaluate the PQ-selected query using true fp32 query-query IP.
            pq_chosen_true[ei]=float(qcache32[lo+pli] @ q)

        agree=float(np.mean(fp_match==pq_match))
        regret=fp_best-pq_chosen_true

        fp_starts=starts_from_match(nav,gt,fp_match,args.result_count)
        pq_starts=starts_from_match(nav,gt,pq_match,args.result_count)
        fpfile=args.out_dir/f"fp16-c{cap}-r{args.result_count}.bin"
        pqfile=args.out_dir/f"pq-c{cap}-r{args.result_count}.bin"
        write_starts(fpfile,fp_starts); write_starts(pqfile,pq_starts)

        meta={
            "capacity":cap,
            "fp16":{
                "query_key_bytes":cap*768*2,
                "plus_anchor_id_bytes":cap*(768*2+4),
                "file":fpfile.name,
                "python_scan_us_per_query":1e6*fp_scan/EVAL_N,
            },
            "pq":{
                "query_key_bytes":cap*qcodes.shape[1],
                "plus_anchor_id_bytes":cap*(qcodes.shape[1]+4),
                "file":pqfile.name,
                "python_scan_us_per_query":1e6*pq_scan/EVAL_N,
            },
            "nearest_match_agreement_fraction":agree,
            "true_similarity_regret":{
                "mean":float(regret.mean()),
                "median":float(np.median(regret)),
                "p95":float(np.quantile(regret,.95)),
                "fraction_zero_or_negative_tol":float(np.mean(regret<=1e-4)),
            },
        }
        result["capacities"][str(cap)]=meta
        print(json.dumps({
            "capacity":cap,
            "agreement":agree,
            "regret_mean":meta["true_similarity_regret"]["mean"],
            "fp16_kib":meta["fp16"]["plus_anchor_id_bytes"]/1024,
            "pq_kib":meta["pq"]["plus_anchor_id_bytes"]/1024,
            "fp16_scan_us":meta["fp16"]["python_scan_us_per_query"],
            "pq_scan_us":meta["pq"]["python_scan_us_per_query"],
        }),flush=True)

    (args.out_dir/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
