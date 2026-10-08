#!/usr/bin/env python3
"""Quantify the routing opportunity that motivates NavHints.

This script is deliberately mechanism-agnostic. It asks two questions using
ordinary DiskANN traces and exact nearest-neighbor ground truth:

1) Junction opportunity: how concentrated is ordinary DiskANN traversal traffic?
2) Destination opportunity: how often do recent successful destinations remain
   useful to later, distinct queries?

No NavHints state or routing policy is used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import struct
from pathlib import Path

import numpy as np


def load_gt(path: Path):
    with path.open("rb") as f:
        raw=f.read(8)
    if len(raw)!=8:
        raise ValueError("bad GT header")
    rows,k=struct.unpack("<II",raw)
    ids_bytes=rows*k*4
    size=path.stat().st_size
    if size not in (8+ids_bytes, 8+2*ids_bytes):
        raise ValueError("unexpected GT size")
    ids=np.fromfile(path,dtype="<u4",count=rows*k,offset=8).reshape(rows,k)
    return ids


def cumulative_curve(counts: collections.Counter[int], corpus_size: int, fractions):
    ordered=np.asarray(sorted(counts.values(), reverse=True),dtype=np.int64)
    total=int(ordered.sum())
    c=np.cumsum(ordered)
    rows=[]
    for frac in fractions:
        n=max(1,min(corpus_size,int(round(frac*corpus_size))))
        # Unobserved corpus vertices simply contribute zero.
        take=min(n,len(c))
        share=float(c[take-1]/total) if take else 0.0
        rows.append({
          "corpus_fraction":float(frac),
          "vertices":int(n),
          "traffic_share":share,
        })
    return rows,total,len(ordered)


def gini_from_sparse_counts(counts, universe):
    x=np.zeros(universe,dtype=np.float64)
    vals=np.asarray(list(counts.values()),dtype=np.float64)
    vals.sort()
    # Avoid materializing/sorting 1M zeros plus counts: zeros precede vals.
    nz=len(vals); z=universe-nz
    if universe<=0 or vals.sum()<=0: return 0.0
    idx=np.arange(z+1,universe+1,dtype=np.float64)
    weighted=np.sum(idx*vals)
    return float((2.0*weighted)/(universe*vals.sum())-(universe+1)/universe)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=Path,required=True)
    ap.add_argument("--gt",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--corpus-size",type=int,default=1_000_000)
    ap.add_argument("--recent-capacity",type=int,default=512)
    ap.add_argument("--skip-prefixes",default="0,5,10,20")
    ap.add_argument("--warmup",type=int,default=4000)
    ap.add_argument("--queries",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)

    records=[json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty traversal trace")

    fracs=(0.0001,0.0005,0.001,0.002,0.005,0.01,0.016,0.02,0.05,0.10)
    skip_prefixes=[int(x) for x in args.skip_prefixes.split(",") if x.strip()]
    if not skip_prefixes or any(x<0 for x in skip_prefixes):
        raise ValueError("invalid skip-prefix list")

    junction_variants={}
    for skip in skip_prefixes:
        visits=collections.Counter()
        support=collections.Counter()
        trace_lens=[]
        for rec in records:
            ids=[int(x) for x in rec["ids"]]
            kept=ids[skip:]
            trace_lens.append(len(kept))
            visits.update(kept)
            support.update(set(kept))
        visit_curve,total_visits,visited_vertices=cumulative_curve(visits,args.corpus_size,fracs)
        support_curve,total_support,_=cumulative_curve(support,args.corpus_size,fracs)
        junction_variants[str(skip)]={
          "skip_first_expansions":skip,
          "total_trace_visits":total_visits,
          "unique_visited_vertices":visited_vertices,
          "mean_trace_length":float(np.mean(trace_lens)),
          "median_trace_length":float(np.median(trace_lens)),
          "visit_gini_over_corpus":gini_from_sparse_counts(visits,args.corpus_size),
          "query_support_gini_over_corpus":gini_from_sparse_counts(support,args.corpus_size),
          "visit_curve":visit_curve,
          "query_support_curve":support_curve,
          "top_vertices_by_query_support":[
            {"vertex":int(v),"queries":int(c),"query_fraction":float(c/len(records)),
             "visits":int(visits[v])}
            for v,c in support.most_common(20)
          ],
        }
    # Preserve the no-skip result under the original key.
    base_junction=junction_variants[str(skip_prefixes[0])]

    gt=load_gt(args.gt)
    if gt.shape[1]<10:
        raise ValueError("need top-10 GT")

    # Recent successful destination opportunity. Use exact top-1 only as the
    # completed-search destination, matching the one-ID-per-query semantics of
    # Recent512. Maintain unique FIFO values as in the deployed bounded set.
    recent=[]
    members=set()
    next_slot=0
    any_top10=[]
    exact_top1=[]
    overlap_counts=[]
    best_ranks=[]
    warm=min(args.warmup,len(gt)-1)
    measured=np.arange(warm,len(gt),dtype=int)

    for qi in range(len(gt)):
        if qi>=warm:
            cur=set(map(int,gt[qi,:10]))
            overlap=[v for v in recent if v in cur]
            any_top10.append(bool(overlap))
            exact_top1.append(int(gt[qi,0]) in members)
            overlap_counts.append(len(overlap))
            if overlap:
                rank={int(v):r+1 for r,v in enumerate(gt[qi,:10])}
                best_ranks.append(min(rank[v] for v in overlap))

        winner=int(gt[qi,0])
        if winner not in members:
            if len(recent)<args.recent_capacity:
                recent.append(winner); members.add(winner)
            else:
                old=recent[next_slot]; members.remove(old)
                recent[next_slot]=winner; members.add(winner)
                next_slot=(next_slot+1)%args.recent_capacity

    # Global result-demand concentration as a complementary view.
    result_counts=collections.Counter(map(int,gt[:,:10].ravel()))
    result_curve,total_result_slots,result_vertices=cumulative_curve(
        result_counts,args.corpus_size,fracs
    )

    def q(a):
        a=np.asarray(a,dtype=np.float64)
        return {str(x):float(np.quantile(a,x)) for x in (0,0.1,0.25,0.5,0.75,0.9,0.99,1)} if len(a) else {}

    repeat_diag=None
    if args.queries is not None:
        with args.queries.open("rb") as f:
            qrows,qdim=struct.unpack("<II",f.read(8))
        if qrows < len(gt):
            raise ValueError("query file shorter than GT")
        qmat=np.memmap(args.queries,dtype="<f4",mode="r",offset=8,shape=(qrows,qdim))
        prior=set(row.tobytes() for row in qmat[:warm])
        seen=set(prior)
        repeated_from_prior=0
        repeated_causally=0
        within_eval_repeats=0
        eval_seen=set()
        for row in qmat[warm:len(gt)]:
            key=row.tobytes()
            if key in prior:
                repeated_from_prior += 1
            if key in seen:
                repeated_causally += 1
            if key in eval_seen:
                within_eval_repeats += 1
            seen.add(key)
            eval_seen.add(key)
        n=max(1,len(gt)-warm)
        repeat_diag={
          "warmup_rows":warm,
          "measured_rows":len(gt)-warm,
          "exact_repeat_from_prior_fraction":repeated_from_prior/n,
          "exact_repeat_causal_fraction":repeated_causally/n,
          "within_measured_repeat_fraction":within_eval_repeats/n,
          "measured_unique_vectors":len(eval_seen),
        }

    result={
      "corpus_size":args.corpus_size,
      "exact_query_repeat_diagnostics":repeat_diag,
      "junctions":{"training_queries":len(records),**base_junction},
      "junctions_by_skip_prefix":junction_variants,
      "destinations":{
        "queries":len(gt),
        "measured_queries":len(measured),
        "recent_capacity":args.recent_capacity,
        "recent_contains_any_exact_top10_fraction":float(np.mean(any_top10)),
        "recent_contains_exact_top1_fraction":float(np.mean(exact_top1)),
        "recent_exact_top10_overlap_count_quantiles":q(overlap_counts),
        "recent_best_exact_rank_quantiles":q(best_ranks),
        "distinct_top10_result_ids":result_vertices,
        "total_top10_result_slots":total_result_slots,
        "result_occurrence_gini_over_corpus":gini_from_sparse_counts(result_counts,args.corpus_size),
        "result_curve":result_curve,
      }
    }
    (args.out/"motivation-opportunity.json").write_text(json.dumps(result,indent=2)+"\n")

    for name,rows,y in (
        ("junction-visit-curve.csv",base_junction["visit_curve"],"traffic_share"),
        ("junction-support-curve.csv",base_junction["query_support_curve"],"traffic_share"),
        ("result-demand-curve.csv",result_curve,"traffic_share"),
    ):
        with (args.out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["corpus_fraction","vertices",y])
            w.writeheader();w.writerows(rows)

    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
