#!/usr/bin/env python3
"""Controlled *pilot* of temporal vs query-conditioned destination locality.

All arms use one pinned NavHints/DiskANN SSD binary, identical learned 16K
junctions, disjoint 5K training history, 4K causal online warm-up and final
1K requests, K=10 and beam=8. No persistent hint writes.

Distinct IDs vs actual RSS: 'budget' is winner-ID payload only. Regional LRU
tracks (id,timestamp) and 512 Vec headers in this correctness-first prototype:
DO NOT interpret as equal total RAM, or make latency claims from these modes.

Baseline must exactly reproduce prior verified FIFO-512 I/O and recall.
"""
import argparse
import json
import os
import re
import statistics
import hashlib
from pathlib import Path
from qualify_navhints_catapult_headtohead import (
    run_one, shape, slice_fbin, TRAIN, WARM, MEASURE, DIM,
)

LS=(40,160,320)
CONFIGS=(
    ("fifo512-k1", "fifo", 512, 1, 1),
    ("lru512-k1", "lru", 512, 1, 1),
    ("region512-n1-k1", "region_lru", 512, 1, 1),
    ("fifo2048-k1", "fifo", 2048, 1, 1),
    ("lru2048-k1", "lru", 2048, 1, 1),
    ("region2048-n1-k1", "region_lru", 2048, 1, 1),
    ("region2048-n4-k1", "region_lru", 2048, 4, 1),
    ("lru8192-k1", "lru", 8192, 1, 1),
    ("region8192-n1-k1", "region_lru", 8192, 1, 1),
    ("region8192-n4-k1", "region_lru", 8192, 4, 1),
    ("region2048-n4-k4", "region_lru", 2048, 4, 4),
    ("region8192-n4-k4", "region_lru", 8192, 4, 4),
)
REFERENCE={
    40:(47.39, 45.935),
    160:(70.17,163.053),
    320:(79.81,322.364),
}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def interp(curve,target):
    points=sorted(
        ({"L":L,**row} for L,row in curve.items()),
        key=lambda x:(x["recall"],x["reads"]))
    if target < points[0]["recall"]-1e-6 or target > points[-1]["recall"]+1e-6:
        return None
    for v in points:
        if abs(target-v["recall"])<1e-6:
            return {"reads":v["reads"],"lower_L":v["L"],"upper_L":v["L"]}
    for a,b in zip(points,points[1:]):
        if a["recall"]<=target<=b["recall"] and b["recall"]>a["recall"]:
            f=(target-a["recall"])/(b["recall"]-a["recall"])
            return {"reads":a["reads"]+f*(b["reads"]-a["reads"]),
                    "lower_L":a["L"],"upper_L":b["L"]}
    return None

def main():
    ap=argparse.ArgumentParser()
    for arg in ("nav-bin","queries","heldout-gt","index-prefix","ivf","work","out"):
        ap.add_argument("--"+arg,type=Path,required=True)
    args=ap.parse_args()
    for key in ("nav_bin","queries","heldout_gt","index_prefix","ivf","work","out"):
        setattr(args,key,getattr(args,key).resolve())
    if shape(args.queries)!=(TRAIN+WARM+MEASURE,DIM):
        raise RuntimeError("not the approved frozen query stream")
    for path in (args.nav_bin,args.heldout_gt,args.ivf):
        if not path.is_file(): raise FileNotFoundError(path)
    args.work.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    heldout=slice_fbin(args.queries,args.work/"heldout5000.fbin",TRAIN,WARM+MEASURE)
    original_affinity=sorted(os.sched_getaffinity(0))
    if len(original_affinity)<4:
        raise RuntimeError("requires four self-hosted CPU cores")
    os.sched_setaffinity(0,set(original_affinity[:4]))

    manifest=json.loads(args.ivf.with_suffix(args.ivf.suffix+".manifest.json").read_text())
    index_bytes=args.ivf.stat().st_size+int(manifest["nlist"])*(64+4)
    if int(manifest["nlist"])!=512:
        raise RuntimeError("unexpected static coarse routing directory")

    results={key:{} for key,_,_,_,_ in CONFIGS}
    stats={}
    try:
        for ix,L in enumerate(LS):
            arms=list(CONFIGS)
            arms=arms[ix%len(arms):]+arms[:ix%len(arms)]
            for key,mode,cap,nprobe,k in arms:
                tag=f"pilot-L{L}-{key}"
                row=run_one(
                    args.nav_bin,tag,heldout,args.heldout_gt,args.index_prefix,
                    L,args.out,mode="nav",ivf=args.ivf,
                    extra_env={
                        "DISKANN_EXPERIENCE_CACHE_CAPACITY":cap,
                        "DISKANN_EXPERIENCE_HUB_CAPACITY":0,
                        "DISKANN_LOCAL_CACHE_MODE":mode,
                        "DISKANN_LOCAL_REGION_NPROBE":nprobe,
                        "DISKANN_LOCAL_RECENT_STARTS":k,
                    })
                log=(args.out/(tag+".log")).read_text()
                match=re.search(
                    rf"NAVHINTS_LOCAL_HISTORY L={L} mode=(\w+) nprobe=(\d+) "
                    r"capacity=(\d+) stored=(\d+) used_regions=(\d+) "
                    r"inserted=(\d+) resident_refreshes=(\d+) evictions=(\d+)",
                    log)
                if match is None:
                    raise RuntimeError(f"{tag}: missing routing-cache statistics")
                measured_mode,measured_probe,measured_cap,stored,used,ins,hits,evictions=match.groups()
                if (measured_mode,int(measured_probe),int(measured_cap))!=(mode,nprobe,cap):
                    raise RuntimeError(f"{tag}: mode mismatch: {match.group(0)}")
                if int(stored)>cap or (mode=="region_lru" and int(used)>512):
                    raise RuntimeError(f"{tag}: cache capacity violation")
                results[key][L]=row
                stats[tag]={
                    "mode":mode,"capacity":cap,"nprobe":nprobe,"starts":k,
                    "entries":int(stored),"nonempty_regions":int(used),
                    "inserts":int(ins),"refreshes":int(hits),
                    "evictions":int(evictions)
                }
                save(args.out/"partial-results.json",results)
                save(args.out/"partial-cache-stats.json",stats)
                if key=="fifo512-k1":
                    reference_recall,reference_reads=REFERENCE[L]
                    # A successful mechanism patch must preserve the exact
                    # original I/O/recall at original parameters.
                    if abs(row["recall"]-reference_recall)>0.10 or abs(row["reads"]-reference_reads)>0.12:
                        raise RuntimeError(f"Old Core control changed at L={L}: {row}")
    finally:
        os.sched_setaffinity(0,set(original_affinity))
    controls=results["fifo512-k1"]
    curve_by_arm={key:{L:results[key][L] for L in LS} for key,_,_,_,_ in CONFIGS}
    comparisons=[]
    for L in LS:
        target=controls[L]["recall"]
        baseline_reads=controls[L]["reads"]
        line={"control_L":L,"control_recall":target,
              "control_reads":baseline_reads,"arms":{}}
        for key,mode,cap,nprobe,k in CONFIGS:
            row=results[key][L]
            x=interp(curve_by_arm[key],target)
            line["arms"][key]={
                "same_L_recall_pp":row["recall"]-target,
                "same_L_delta_reads":row["reads"]-baseline_reads,
                "same_L_read_saving_percent":100*(1-row["reads"]/baseline_reads),
                "matched_recall":None if x is None else {
                    **x, "read_saving_percent":100*(1-x["reads"]/baseline_reads)
                }
            }
        comparisons.append(line)
    summary={
        "purpose":"Exploratory ablation of global FIFO, global LRU, and learned-region-conditioned LRU. NOT a validated end-to-end speedup.",
        "workload":"PubMed1M/MedCPT 1M 768d IP, first 5K static-only history, heldout 4K warmup plus 1K measured",
        "backend":"pinned Microsoft DiskANN3, original 16K Hint-IVF directory",
        "scoring":"K=10, beam=8, real NVMe, native logical SSD reads, recall matched by interpolation where available",
        "memory_contract":"routing static bytes plus cache ID payload only. Regional prototype also stores per-ID timestamps and Vec metadata: real RAM is higher than 4B per destination.",
        "region_key":"nearest already-PQ-scored Hint-IVF coarse medoid; optionally top four query regions; globally budgeted demand-weighted LRU winner histories",
        "pilot":"3 L widths, one deterministic heldout stream, no hash seeds or repetition for NavHints; preliminary findings only",
        "static_hint_bytes":index_bytes,
        "configs":[{"key":key,"mode":mode,"id_capacity":cap,"value_id_payload_bytes":cap*4,
                    "region_nprobe":nprobe,"additional_destination_starts":k}
                   for key,mode,cap,nprobe,k in CONFIGS],
        "results":results,"cache_stats":stats,
        "comparison_against_original_fifo512":comparisons,
        "ivf_sha256":hashlib.sha256(args.ivf.read_bytes()).hexdigest(),
    }
    save(args.out/"navhints-regional-history-pilot.json",summary)
    print("NAVHINTS_REGIONAL_HISTORY_PILOT_COMPLETE",flush=True)
    for row in comparisons:
        print(f"L={row['control_L']} recall={row['control_recall']:.2f} control_io={row['control_reads']:.3f}",flush=True)
        for k,v in row["arms"].items():
            m=v["matched_recall"]
            txt="unavailable" if m is None else f"{m['read_saving_percent']:+.3f}%"
            print(f"  {k:22} same_L={v['same_L_delta_reads']:+.3f} recall_pp={v['same_L_recall_pp']:+.3f} matched_saving={txt}",flush=True)

if __name__=="__main__":
    main()
