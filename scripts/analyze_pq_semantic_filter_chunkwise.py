#!/usr/bin/env python3
"""Tighter certified PQ semantic filter using per-chunk residual radii.

For PQ chunks j, store epsilon[x,j] = ||x_j - xhat_j||_2. Then

  q.x <= q.xhat + sum_j ||q_j||_2 * epsilon[x,j].

This remains a deterministic one-sided certificate but is never looser than
the single global Cauchy ball. Radii are stored as upward-rounded float16 in
this screen, so the memory cost is 2 * npoints * nchunks bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_pq_semantic_filter import (
    LS, Graph, fbin_memmap, gt_ids, load_pq, lut_for_query, pq_scores, rate
)


def compute_chunk_eps(base, pivots, chunk_offsets, codes, cache: Path):
    n, dim = base.shape
    m = codes.shape[1]
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        if arr.shape == (n, m) and arr.dtype == np.float16:
            print(f"loaded chunk epsilon cache {cache} shape={arr.shape}", flush=True)
            return arr

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp.npy")
    if tmp.exists():
        tmp.unlink()
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, m))
    block = 8192
    for s in range(0, n, block):
        e = min(n, s + block)
        xb = np.asarray(base[s:e], dtype=np.float32)
        cb = codes[s:e]
        for j in range(m):
            a, b = int(chunk_offsets[j]), int(chunk_offsets[j+1])
            recon = pivots[cb[:, j], a:b]
            d = xb[:, a:b] - recon
            norm = np.sqrt(np.einsum("ij,ij->i", d, d, optimize=True)).astype(np.float32)
            h = norm.astype(np.float16)
            low = h.astype(np.float32) < norm
            if np.any(low):
                h[low] = np.nextafter(h[low], np.float16(np.inf))
            out[s:e, j] = h
        if s == 0 or e % 100000 < block or e == n:
            print(f"chunk_residuals={e}/{n}", flush=True)
    out.flush()
    del out
    tmp.replace(cache)
    return np.load(cache, mmap_mode="r")


def q_chunk_norms(q, chunk_offsets):
    out = np.empty(len(chunk_offsets)-1, dtype=np.float32)
    for j in range(len(out)):
        a,b=int(chunk_offsets[j]),int(chunk_offsets[j+1])
        out[j]=np.linalg.norm(q[a:b])
    return out


def uncertainty(ids, qnorms, eps_chunk):
    ids=np.asarray(ids,dtype=np.int64)
    if ids.size==0:
        return np.empty(0,dtype=np.float32)
    e=np.asarray(eps_chunk[ids],dtype=np.float32)
    return e @ qnorms


def main():
    ap=argparse.ArgumentParser()
    for name in ("base","queries","gt5000","pq-pivots","pq-codes","disk-index","trace-dir","out","chunk-epsilon-cache"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--query-offset",type=int,default=9000)
    ap.add_argument("--gt-offset",type=int,default=4000)
    ap.add_argument("--k",type=int,default=10)
    args=ap.parse_args()

    base=fbin_memmap(args.base.resolve())
    qall=fbin_memmap(args.queries.resolve())
    queries=np.asarray(qall[args.query_offset:args.query_offset+1000],dtype=np.float32)
    gtall=gt_ids(args.gt5000.resolve())
    gt=gtall[args.gt_offset:args.gt_offset+len(queries)]
    pivots,chunk_offsets,codes=load_pq(args.pq_pivots.resolve(),args.pq_codes.resolve())
    eps=compute_chunk_eps(base,pivots,chunk_offsets,codes,args.chunk_epsilon_cache.resolve())
    graph=Graph(args.disk_index.resolve())
    if graph.h["n"]!=base.shape[0]:
        raise ValueError("graph/base mismatch")

    m=codes.shape[1]
    result={
        "certificate":{
            "formula":"upper(q,x)=q dot xhat + sum_j ||q_j|| * epsilon[x,j]",
            "radii_storage":"upward-rounded float16 per PQ chunk",
            "guarantee":"upper(q,x) >= exact inner product q dot x",
        },
        "pq":{
            "chunks":int(m),
            "centers":int(pivots.shape[0]),
            "dimension":int(pivots.shape[1]),
            "chunk_radii_bytes_for_1M":int(eps.nbytes),
            "bytes_per_vector":int(m*2),
        },
        "by_L":{},
    }
    tol=3e-4
    max_violation=0.0
    total_topk_neighbor_violations=0

    for L in LS:
        trace_path=args.trace_dir.resolve()/f"heldout.L{L}.jsonl"
        recs=[json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()]
        if len(recs)!=len(queries):
            raise ValueError(f"L{L}: trace/query mismatch")

        c={
            "expansions":0,
            "point_final_reject":0,
            "point_current_eligible":0,
            "point_current_reject":0,
            "neighbors_total":0,
            "neighbors_final_rejected":0,
            "neighbor_sets":0,
            "neighbor_all_final_reject":0,
            "neighbor_current_eligible":0,
            "neighbor_all_current_reject":0,
            "topk_neighbor_exposures":0,
            "certified_all_with_topk_neighbor":0,
        }
        stage=[{"n":0,"point":0,"allnbr":0} for _ in range(4)]
        ratios=[]

        for qi,rec in enumerate(recs):
            q=queries[qi]
            lut=lut_for_query(q,pivots,chunk_offsets)
            qn=q_chunk_norms(q,chunk_offsets)
            qnorm=float(np.linalg.norm(q))
            truth=gt[qi,:args.k].astype(np.int64)
            truth_scores=np.asarray(base[truth],dtype=np.float32)@q
            tau_final=float(np.min(truth_scores))
            truth_set=set(int(x) for x in truth)

            ids=[int(x) for x in rec["ids"]]
            exact_seen=[]
            for pos,x in enumerate(ids):
                quart=min(3,(4*pos)//max(1,len(ids)))
                c["expansions"]+=1
                stage[quart]["n"]+=1
                sx=float(pq_scores([x],lut,codes)[0])
                ux_unc=float(uncertainty([x],qn,eps)[0])
                ux=sx+ux_unc
                exact_x=float(np.dot(np.asarray(base[x],dtype=np.float32),q))
                max_violation=max(max_violation,exact_x-ux)

                # Quantify how much tighter chunkwise is than a global Cauchy ball
                # built from the same per-chunk radii.
                er=np.asarray(eps[x],dtype=np.float32)
                global_unc=qnorm*float(np.linalg.norm(er))
                if global_unc>0:
                    ratios.append(ux_unc/global_unc)

                if ux<tau_final-tol:
                    c["point_final_reject"]+=1
                    stage[quart]["point"]+=1

                if len(exact_seen)>=args.k:
                    tau_cur=float(np.partition(np.asarray(exact_seen),-args.k)[-args.k])
                    c["point_current_eligible"]+=1
                    if ux<tau_cur-tol:
                        c["point_current_reject"]+=1
                else:
                    tau_cur=None

                nbrs=graph.neighbors(x).astype(np.int64,copy=False)
                c["neighbor_sets"]+=1
                c["neighbors_total"]+=int(nbrs.size)
                if nbrs.size:
                    sn=pq_scores(nbrs,lut,codes).astype(np.float64)
                    un=sn+uncertainty(nbrs,qn,eps).astype(np.float64)
                    rf=un<tau_final-tol
                    c["neighbors_final_rejected"]+=int(rf.sum())
                    allf=bool(np.all(rf))
                    if allf:
                        c["neighbor_all_final_reject"]+=1
                        stage[quart]["allnbr"]+=1
                    has_truth=any(int(y) in truth_set for y in nbrs)
                    if has_truth:
                        c["topk_neighbor_exposures"]+=1
                        if allf:
                            c["certified_all_with_topk_neighbor"]+=1
                    if tau_cur is not None:
                        c["neighbor_current_eligible"]+=1
                        if bool(np.all(un<tau_cur-tol)):
                            c["neighbor_all_current_reject"]+=1

                exact_seen.append(exact_x)

        total_topk_neighbor_violations += c["certified_all_with_topk_neighbor"]
        result["by_L"][str(L)]={
            "counts":c,
            "rates":{
                "point_final_reject_fraction":rate(c["point_final_reject"],c["expansions"]),
                "point_current_reject_fraction_after_k_validated":rate(c["point_current_reject"],c["point_current_eligible"]),
                "individual_neighbor_final_reject_fraction":rate(c["neighbors_final_rejected"],c["neighbors_total"]),
                "neighbor_set_all_final_reject_fraction":rate(c["neighbor_all_final_reject"],c["neighbor_sets"]),
                "neighbor_set_all_current_reject_fraction_after_k_validated":rate(c["neighbor_all_current_reject"],c["neighbor_current_eligible"]),
            },
            "uncertainty":{
                "median_chunk_over_global":float(np.median(ratios)) if ratios else 0.0,
                "mean_chunk_over_global":float(np.mean(ratios)) if ratios else 0.0,
                "p95_chunk_over_global":float(np.quantile(ratios,.95)) if ratios else 0.0,
            },
            "stage_quartiles":[
                {
                    "quartile":i+1,
                    "expansions":d["n"],
                    "point_final_reject_fraction":rate(d["point"],d["n"]),
                    "neighbor_set_all_final_reject_fraction":rate(d["allnbr"],d["n"]),
                } for i,d in enumerate(stage)
            ],
        }

    result["safety_checks"]={
        "max_exact_minus_upper_bound":float(max_violation),
        "neighbor_all_certificate_true_topk_violations":int(total_topk_neighbor_violations),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
