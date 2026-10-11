#!/usr/bin/env python3
"""Controlled SAME DISK SSD rerun: legacy Catapult vs author-aligned Catapult.

The original source is never changed. Two separate binaries differ only in
Catapult RNG/bitpacking, medoid insertion and timing boundary. The graph,
SSD, PQ, 8bitx40 ID payload, queries, recall GT and beam widths are identical.

Use sequential, chronologically ordered 5K online TRAIN for LRU causality,
then freeze snapshot and evaluate last 5K with four native DiskANN threads.
This does NOT claim to reproduce the original in-memory ANN CPU performance.
"""
import argparse
import json
import os
import statistics
from pathlib import Path

from qualify_frozen_catapult_heldout import (
    result_rows, slice_fbin, fbin_shape, snapshot_entries, save,
)
import subprocess

K=10
WIDTHS=(20,40,80,160,320)
SEEDS=(0,1,2)
BEAM=8
HASHES=8
CAP=80


def cfg(queries, gt, index, width, threads):
    return {
        "search_directories":[str(queries.parent)],
        "jobs":[{"type":"disk-index","content":{
           "source":{
             "disk-index-source":"Load","data_type":"float32",
             "load_path":str(index)},
           "search_phase":{
             "queries":str(queries),"groundtruth":str(gt),
             "search_list":[width],"beam_width":BEAM,
             "recall_at":K,"num_threads":threads,
             "is_flat_search":False,"distance":"inner_product",
             "vector_filters_file":None,"num_nodes_to_cache":None,
             "search_io_limit":None,"post_processor":None,
           },
        }}],
    }

def run(binary,out,tag,query,gt,index,L,threads,seed=None,train=False,
        snapshot_load=None,snapshot_dump=None):
    config_file=out/f"{tag}-input.json"
    result_file=out/f"{tag}-output.json"
    save(config_file,cfg(query,gt,index,L,threads))
    env=os.environ.copy()
    for k in list(env):
        if k.startswith("DISKANN_CATAPULT_") or k in (
            "DISKANN_PAPER_CATAPULT",
            "DISKANN_SKIP_RECALL",
            "DISKANN_IP_PORTAL_ROUTER_FILE",
            "DISKANN_IP_PORTAL_NPROBE",
            "DISKANN_WAYPOINT_CACHE_FILE",
            "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"):
            env.pop(k,None)
    if train:
        env["DISKANN_SKIP_RECALL"]="1"
    if seed is not None:
        env["DISKANN_PAPER_CATAPULT"]="1"
        env["DISKANN_CATAPULT_HASHES"]=str(HASHES)
        env["DISKANN_CATAPULT_CAPACITY"]=str(CAP)
        env["DISKANN_CATAPULT_SEED"]=str(seed)
    if snapshot_load is not None:
        env["DISKANN_CATAPULT_SNAPSHOT_LOAD"]=str(snapshot_load)
        env["DISKANN_CATAPULT_FREEZE"]="1"
    if snapshot_dump is not None:
        env["DISKANN_CATAPULT_SNAPSHOT_DUMP"]=str(snapshot_dump)

    with (out/f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary),"run","--input-file",str(config_file),
             "--output-file",str(result_file)],
            check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=result_rows(json.loads(result_file.read_text()))
    if len(rows)!=1 or int(rows[0]["search_l"])!=L:
        raise ValueError(f"{tag}: bad native result rows")
    row=rows[0]
    if train:
        assert float(row["recall"])==-1.0
    else:
        assert 0<=float(row["recall"])<=100, f"{tag}: invalid recall {row['recall']}"
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--legacy-bin",type=Path,required=True)
    ap.add_argument("--author-bin",type=Path,required=True)
    ap.add_argument("--queries",type=Path,required=True)
    ap.add_argument("--train-gt",type=Path,required=True)
    ap.add_argument("--eval-gt",type=Path,required=True)
    ap.add_argument("--index-prefix",type=Path,required=True)
    ap.add_argument("--work",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    for key in ("legacy_bin","author_bin","queries","train_gt","eval_gt",
                "index_prefix","work","out"):
        setattr(args,key,getattr(args,key).resolve())
    args.work.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    assert all(p.is_file() for p in (
        args.legacy_bin,args.author_bin,args.queries,args.train_gt,args.eval_gt))
    assert fbin_shape(args.queries)==(10000,768)
    train=slice_fbin(args.queries,args.work/"train-prefix.fbin",0,5000)
    held=slice_fbin(args.queries,args.work/"eval-suffix.fbin",5000,5000)
    allowed=sorted(os.sched_getaffinity(0))
    assert len(allowed)>=4
    os.sched_setaffinity(0,set(allowed[:4]))

    # This action is run with exclusive speed-device.lock around the entire
    # binary/replay sequence by run_catapult_fidelity_ssd_ci.sh.
    # Always compare the same L, seed, full graph and matched frozen query IDs.
    rows=[]
    try:
        for L in WIDTHS:
            baseline=run(args.legacy_bin,args.out,f"baseline-L{L}",
                held,args.eval_gt,args.index_prefix,L,4)
            rows.append({"L":L,"variant":"baseline","seed":None,
                         "training":"none","data":baseline})
            print(f"AUTHOR_CATAPULT_PARITY_BASELINE L={L} "
                  f"recall={baseline['recall']} IO={baseline['mean_ios']}",flush=True)

            for seed in SEEDS:
                # Round-trip legacy and author-aligned sources on identical
                # train prefix. Alternating order reduces device drift.
                order=("legacy","author_aligned") if seed%2==0 else (
                    "author_aligned","legacy")
                for variant in order:
                    binary=args.legacy_bin if variant=="legacy" else args.author_bin
                    label=f"{variant}-L{L}-seed{seed}"
                    snapshot=args.work/(label+".snapshot")
                    before=run(binary,args.out,"train-"+label,
                        train,args.train_gt,args.index_prefix,L,1,
                        seed=seed,train=True,snapshot_dump=snapshot)
                    count=snapshot_entries(snapshot)
                    assert 0<count<=(1<<HASHES)*CAP,(label,count)
                    measured=run(binary,args.out,"eval-"+label,
                        held,args.eval_gt,args.index_prefix,L,4,
                        seed=seed,snapshot_load=snapshot)
                    rows.append({"L":L,"variant":variant,"seed":seed,
                                 "training":{
                                     "queries":5000,"threads":1,
                                     "snapshot_entries":count,
                                     "updates_during_eval":False
                                 },"data":measured})
                    save(args.out/"rows.partial.json",rows)
                    print(f"AUTHOR_CATAPULT_PARITY_PAIR {label} "
                          f"IO={measured['mean_ios']} "
                          f"recall={measured['recall']} "
                          f"latency={measured['mean_latency']}",
                          flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))
    save(args.out/"rows.json",rows)

    summary={}
    for L in WIDTHS:
        summary[str(L)]={}
        for variant in ("baseline","legacy","author_aligned"):
            s=[r["data"] for r in rows if r["L"]==L and r["variant"]==variant]
            assert len(s)==(1 if variant=="baseline" else len(SEEDS))
            summary[str(L)][variant]={
                metric:statistics.mean(float(x[metric]) for x in s)
                for metric in ("recall","mean_ios","mean_latency",
                               "mean_io_time","mean_cpu_time","qps")
            }
    out={
        "dataset":"PubMed1M / MedRAG-Zipf first 5K train, next 5K eval",
        "description":"Author-aligned versus legacy Catapult, identical SSD graph/IO/PQ, and original author's LRU structure, seeds (0,1,2)",
        "source_kind":"adaptations running in pinned Microsoft DiskANN, NOT author's native in-memory system",
        "same_auxiliary_ID_budget":"256 buckets × 80 slots × 4B = 81,920B plus 24,576B hyperplanes = 106,496B, close to 102,940B NavHints Core (both before container metadata)",
        "search":"Recall@10, L=20/40/80/160/320, beam=8, 1-thread causal training, 4-thread frozen evaluation",
        "changes":"StdNormal/MSB hash seed parity; medoid admission to LRU; include update in query latency",
        "limitations":"Frozen eval omits updates by design; exact author library uses an in-memory graph, not this SSD",
        "summary":summary,
        "success":True,
    }
    save(args.out/"catapult-fidelity-ssd-summary.json",out)
    print("CATAPULT_FIDELITY_SSD_3SEED_MATCHED_L_COMPLETED",flush=True)
    print(json.dumps(out["summary"],indent=2),flush=True)
if __name__=="__main__":main()
