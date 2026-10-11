#!/usr/bin/env python3
"""Same-SSD, *accounted-payload-equal* allocation sweep for NavHints.

At 512 fixed IVF cells, packed 16K hint map + Recent512 occupies 102,940
bytes of measured auxiliary payload. Shrinking static hints by (C - 512)
records leaves room for C recent-result IDs at exactly the same total:
B + C = 16,512. Each candidate IVF is independently clustered from
its top-B navigation landmark IDs with identical training history.

The prototype still stores a HashSet, temporary scores, allocator/container
metadata, and prepacked coarse PQ in memory. Equal *payload* here does not
establish equal RSS; a later compact implementation must make that real.

Only RAM-only Core is tested (hub_capacity=0, no persistent writes).
One deterministic PubMed held-out 4k+1k replay; not independent samples.
"""
import argparse
import json
import hashlib
import os
import re
import statistics
from pathlib import Path

from qualify_navhints_catapult_headtohead import (
    run_one, slice_fbin, shape, TRAIN, WARM, MEASURE, DIM
)

LS=(20,40,80,120,160,200,240,280,320)
# (label, junctions B, online destination ID capacity C, injected starts K)
ARMS=(
    ("original-16000-512-k1", 16000, 512, 1),
    ("original-16000-512-k4", 16000, 512, 4),
    ("equal-15488-1024-k4", 15488, 1024, 4),
    ("equal-14464-2048-k2", 14464, 2048, 2),
    ("equal-14464-2048-k4", 14464, 2048, 4),
    ("equal-14464-2048-k8", 14464, 2048, 8),
    ("equal-12416-4096-k4", 12416, 4096, 4),
    ("equal-8320-8192-k4", 8320, 8192, 4),
    ("extra-16000-2048-k4", 16000, 2048, 4),
)
N_LIST=512
PRIMARY_REFS={
    40:(47.38999938964844,45.935),
    80:(58.720001220703125,84.021),
    160:(70.16999816894531,163.053),
    320:(79.80999755859375,322.364),
}
ALLOWED_PAYLOAD=102940

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def frontier(curve):
    points=sorted(
        ({"L":int(L),**v} for L,v in curve.items()),
        key=lambda x:(x["recall"],x["reads"]))
    result=[]
    for p in points:
        if result and p["recall"] <= result[-1]["recall"]+1e-7:
            if p["reads"] < result[-1]["reads"]:
                result[-1]=p
        else: result.append(p)
    return result

def interp(curve,target):
    points=frontier(curve)
    if not points or target < points[0]["recall"]-1e-7 or target > points[-1]["recall"]+1e-7:
        return None
    for p in points:
        if abs(target-p["recall"]) <1e-7:
            return {"reads":p["reads"],"latency_us":p["latency_us"],"lower_L":p["L"],"upper_L":p["L"]}
    for lo,hi in zip(points,points[1:]):
        if lo["recall"] < target < hi["recall"]:
            f=(target-lo["recall"])/(hi["recall"]-lo["recall"])
            return {"reads":lo["reads"]+f*(hi["reads"]-lo["reads"]),
                    "latency_us":lo["latency_us"]+f*(hi["latency_us"]-lo["latency_us"]),
                    "lower_L":lo["L"],"upper_L":hi["L"]}
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("nav-bin","queries","heldout-gt","index-prefix","ivf-dir","work","out"):
        ap.add_argument("--"+arg,type=Path,required=True)
    a=ap.parse_args()
    for key in ("nav_bin","queries","heldout_gt","index_prefix","ivf_dir","work","out"):
        setattr(a,key,getattr(a,key).resolve())
    if shape(a.queries)!=(TRAIN+WARM+MEASURE,DIM):
        raise ValueError("unexpected frozen query split")
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    held=slice_fbin(a.queries,a.work/"heldout5000.fbin",TRAIN,WARM+MEASURE)
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<4:
        raise RuntimeError("need at least four self-hosted CPU cores")
    os.sched_setaffinity(0,set(allowed[:4]))
    configs={}
    for tag,B,C,K in ARMS:
        file=a.ivf_dir/f"hints-b{B}-nlist512-spherical.bin"
        manifest_path=file.with_suffix(file.suffix+".manifest.json")
        if not file.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"{tag} IVF or manifest not built")
        manifest=json.loads(manifest_path.read_text())
        if int(manifest["nlist"])!=N_LIST or int(manifest["landmark_ids"])!=B:
            raise ValueError(f"{tag} incorrect IVFs: {manifest}")
        static_bytes=file.stat().st_size+N_LIST*(64+4)
        total=static_bytes+C*4
        if tag.startswith("equal-") or tag.startswith("original-"):
            if total!=ALLOWED_PAYLOAD:
                raise ValueError(f"{tag}: unequal payload total={total} instead of {ALLOWED_PAYLOAD}")
        elif tag.startswith("extra-"):
            if total<=ALLOWED_PAYLOAD:
                raise ValueError(f"{tag}: missing extra budget")
        else:
            raise ValueError(f"invalid arm name {tag}")
        configs[tag]={"B":B,"C":C,"K":K,
                      "static_bytes":static_bytes,"winner_payload_bytes":C*4,
                      "total_auxiliary_payload_bytes":total,
                      "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "ivf_sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    results={tag:{} for tag,*_ in ARMS}
    stats={}
    try:
        for ix,L in enumerate(LS):
            arms=list(ARMS)
            arms=arms[ix%len(arms):]+arms[:ix%len(arms)]
            for tag,B,C,K in arms:
                file=a.ivf_dir/f"hints-b{B}-nlist512-spherical.bin"
                name=f"tradeoff-L{L}-{tag}"
                print(f"TRADEOFF L={L} arm={tag} total_payload={configs[tag]['total_auxiliary_payload_bytes']}",flush=True)
                row=run_one(
                    a.nav_bin,name,held,a.heldout_gt,a.index_prefix,L,a.out,
                    mode="nav",ivf=file,
                    extra_env={
                        "DISKANN_EXPERIENCE_CACHE_CAPACITY":C,
                        "DISKANN_EXPERIENCE_HUB_CAPACITY":0,
                        "DISKANN_LOCAL_CACHE_MODE":"fifo",
                        "DISKANN_LOCAL_REGION_NPROBE":1,
                        "DISKANN_LOCAL_RECENT_STARTS":K,
                    })
                raw=(a.out/(name+".log")).read_text()
                m=re.search(
                    rf"EXPERIENCE_STATS L={L} cache_capacity=(\d+) hub_capacity=(\d+).*? "
                    r"cache_inserts=(\d+) cache_skips=(\d+) cache_evictions=(\d+).*?"
                    r" eval_writes=(\d+)",raw)
                if not m:
                    raise RuntimeError(f"{name}: missing cache/eval-write proof")
                capacity,hub,ins,hits,evict,writes=map(int,m.groups())
                if (capacity,hub,writes)!=(C,0,0):
                    raise RuntimeError(f"{name}: unexpected Core state changes {m.groups()}")
                resident=min(ins,C)
                stats[name]={"capacity":capacity,"insertions":ins,
                    "duplicate_residents":hits,"evictions":evict,
                    "resident":resident}
                if tag=="original-16000-512-k1" and L in PRIMARY_REFS:
                    want_recall,want_reads=PRIMARY_REFS[L]
                    if abs(row["recall"]-want_recall)>0.10 or abs(row["reads"]-want_reads)>0.12:
                        raise RuntimeError(f"Original Core changed at L={L}: {row}")
                results[tag][str(L)]=row
                save(a.out/"partial-results.json",results)
                save(a.out/"partial-stats.json",stats)
    finally:
        os.sched_setaffinity(0,set(allowed))
    target=results["original-16000-512-k1"]
    matched={}
    for arm,_,_,_ in ARMS:
        matched[arm]={}
        for L in LS:
            baseline=target[str(L)]
            x=interp(results[arm],baseline["recall"])
            matched[arm][str(L)]={
                "target_recall":baseline["recall"],
                "matched":x,
                "core_reads":baseline["reads"],
                "reads_saved_percent":None if x is None else 100*(1-x["reads"]/baseline["reads"]),
            }
    out={
        "scope":"same-SSD, fixed accounted routing payload, one deterministic PubMed replay",
        "caveat":"equal payload, not equal process RSS, temporary PQ buffers or HashSet metadata; one replay only; no latency significance",
        "dataset":"PubMed1M/MedCPT, frozen MedRAG-Zipf 5K disjoint training +4K causal warmup+1K measured",
        "junction_selection":"top-B unique vertices from same frozen 5K traversal history, re-clustered to 512 medoids using same seed; no new training queries",
        "search":"K=10, beam=8, one query at a time, 4 engine search threads, PQ routing computation included in timed query",
        "widths":list(LS),"accounted_byte_budget":ALLOWED_PAYLOAD,
        "configs":configs,"results":results,"stats":stats,
        "matched_recall_against_original":matched,
    }
    save(a.out/"navhints-memory-tradeoff.json",out)
    print("NAVHINTS_FIXED_PAYLOAD_TRADEOFF_PASS",flush=True)
    for L in LS:
        print(f"L={L} original recall={target[str(L)]['recall']:.3f} io={target[str(L)]['reads']:.3f}",flush=True)
        for arm,*_ in ARMS:
            y=matched[arm][str(L)]
            saving=y["reads_saved_percent"]
            if saving is not None:
                print(f"  {arm:26} saving={saving:+.3f}% matched_io={y['matched']['reads']:.3f}",flush=True)
            else:
                print(f"  {arm:26} not within measured recall overlap",flush=True)

if __name__=="__main__":
    main()
