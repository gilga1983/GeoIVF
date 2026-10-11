#!/usr/bin/env python3
"""Ablate coarse query-conditioned history for the same *unchanged* 16K router.

All arms are one pinned Microsoft DiskANN SSD graph and 5K/4K/1K causal
PubMed/MedCPT stream. Every arm keeps 2048 destination slots and introduces
its 4 best local PQ-scored starts. Regional arms have 1 or 4 probed coarse
cells, with optional cluster-based 512->16/32/64/128/256 macro lookup.

Compare logical reads and recall at same L and interpolated matched recall,
and report native CPU residuals (not independent CPU-cycle measurements).
The mapping adds ~1040B and regional Vec/timestamps add more; total process
RSS is NOT held constant in this mechanism diagnostic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
from qualify_navhints_catapult_headtohead import (
    run_one, shape, slice_fbin, TRAIN, WARM, MEASURE, DIM
)

LS=(40,80,160,320)
MODES=(
    ("global2048-k4","fifo",None,1),
    ("region512-n1-k4","region_lru",None,1),
    ("region512-n4-k4","region_lru",None,4),
    ("macro16-n1-k4","region_lru",16,1),
    ("macro16-n4-k4","region_lru",16,4),
    ("macro32-n1-k4","region_lru",32,1),
    ("macro32-n4-k4","region_lru",32,4),
    ("macro64-n1-k4","region_lru",64,1),
    ("macro64-n4-k4","region_lru",64,4),
    ("macro128-n1-k4","region_lru",128,1),
    ("macro128-n4-k4","region_lru",128,4),
    ("macro256-n1-k4","region_lru",256,1),
    ("macro256-n4-k4","region_lru",256,4),
)
REF={
    40:(49.279998779296875,44.230),
    80:(60.33000183105469,83.314),
    160:(71.16999816894531,162.630),
    320:(80.19000244140625,322.072),
}

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")

def frontier(curve):
    pts=sorted(
        ({"L":L,**entry} for L,entry in curve.items()),
        key=lambda r:(r["recall"],r["reads"]),
    )
    out=[]
    for p in pts:
        if out and p["recall"]<=out[-1]["recall"]+1e-7:
            if p["reads"]<out[-1]["reads"]:out[-1]=p
        else:out.append(p)
    return out

def interp(curve,target):
    pts=frontier(curve)
    if target<pts[0]["recall"]-1e-7 or target>pts[-1]["recall"]+1e-7:
        return None
    for x in pts:
        if abs(target-x["recall"])<1e-7:
            return {"reads":x["reads"],"lower_L":x["L"],"upper_L":x["L"]}
    for lo,hi in zip(pts,pts[1:]):
        if lo["recall"]<target<hi["recall"]:
            a=(target-lo["recall"])/(hi["recall"]-lo["recall"])
            return {"reads":lo["reads"]+a*(hi["reads"]-lo["reads"]),
                    "lower_L":lo["L"],"upper_L":hi["L"]}
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("nav-bin","queries","heldout-gt","index-prefix","ivf","macro-dir","work","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    for k in ("nav_bin","queries","heldout_gt","index_prefix","ivf","macro_dir","work","out"):
        setattr(a,k,getattr(a,k).resolve())
    if shape(a.queries)!=(TRAIN+WARM+MEASURE,DIM):
        raise RuntimeError("expected original frozen causal PubMed query sequence")
    for p in (a.ivf,a.nav_bin,a.heldout_gt):
        if not p.is_file():raise FileNotFoundError(p)
    manifest=json.loads(
        a.ivf.with_suffix(a.ivf.suffix+".manifest.json").read_text())
    if int(manifest["nlist"])!=512 or int(manifest["landmark_ids"])!=16000:
        raise RuntimeError("must retain original 16K static router")
    mm=json.loads((a.macro_dir/"macro.manifest.json").read_text())
    original_ivf_hash=hashlib.sha256(a.ivf.read_bytes()).hexdigest()
    if mm["original_ivf_sha256"]!=original_ivf_hash:
        raise RuntimeError("macro grouping not trained from this static router")
    map_files={}
    for P in (16,32,64,128,256):
        f=a.macro_dir/f"macro-n{P}.bin"
        assert f.is_file()
        meta=mm["macro_maps"][str(P)]
        if hashlib.sha256(f.read_bytes()).hexdigest()!=meta["sha256"]:
            raise RuntimeError(f"tampered macro grouping {P}")
        map_files[P]=f

    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    held=slice_fbin(a.queries,a.work/"heldout5000.fbin",TRAIN,WARM+MEASURE)
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<4:raise RuntimeError("self-hosted runner with 4 CPU cores required")
    os.sched_setaffinity(0,set(allowed[:4]))
    results={name:{} for name,*_ in MODES}
    diag={}
    try:
        for i,L in enumerate(LS):
            configs=list(MODES)
            configs=configs[i%len(configs):]+configs[:i%len(configs)]
            for name,policy,P,nprobe in configs:
                tag=f"macro-L{L}-{name}"
                overrides={
                    "DISKANN_EXPERIENCE_CACHE_CAPACITY":"2048",
                    "DISKANN_EXPERIENCE_HUB_CAPACITY":"0",
                    "DISKANN_LOCAL_CACHE_MODE":policy,
                    "DISKANN_LOCAL_REGION_NPROBE":nprobe,
                    "DISKANN_LOCAL_RECENT_STARTS":"4",
                }
                if P is not None:
                    overrides["DISKANN_LOCAL_MACRO_MAP_FILE"]=map_files[P]
                row=run_one(a.nav_bin,tag,held,a.heldout_gt,a.index_prefix,
                            L,a.out,mode="nav",ivf=a.ivf,extra_env=overrides)
                log=(a.out/(tag+".log")).read_text()
                rgx=rf"NAVHINTS_LOCAL_HISTORY L={L} mode=(\w+) nprobe=(\d+) capacity=(\d+) stored=(\d+) used_regions=(\d+) inserted=(\d+) resident_refreshes=(\d+) evictions=(\d+)"
                m=re.search(rgx,log)
                if not m:
                    raise RuntimeError(f"{tag} no valid cache stats")
                mmode,mprobe,capacity,stored,used,ins,hits,evict=m.groups()
                if mmode!=policy or int(mprobe)!=nprobe or int(capacity)!=2048:
                    raise RuntimeError(f"{tag} config ignored")
                if int(stored)>2048 or (P is not None and int(used)>P):
                    raise RuntimeError(f"{tag} region/slot overflow")
                if P is not None and f"NAVHINTS_MACRO_MAP ncoarse=512 nmacro={P}" not in log:
                    raise RuntimeError(f"{tag} no macro-map load proof")
                if name=="global2048-k4":
                    recall,reads=REF[L]
                    if abs(row["recall"]-recall)>0.11 or abs(row["reads"]-reads)>0.11:
                        raise RuntimeError(f"original FIFO2048 control changed at L={L}: {row}")
                diag[tag]={"mode":mmode,"nprobe":int(mprobe),"slots":int(stored),
                    "nonempty_history_buckets":int(used),"insertions":int(ins),
                    "reuses":int(hits),"evictions":int(evict)}
                results[name][L]=row
                save(a.out/"partial-results.json",results)
                save(a.out/"partial-stats.json",diag)
    finally:
        os.sched_setaffinity(0,set(allowed))

    reference=results["global2048-k4"]
    comparisons={}
    for name,*_ in MODES:
        comparisons[name]={}
        for L in LS:
            ref=reference[L]; r=results[name][L]; aligned=interp(results[name],ref["recall"])
            comparisons[name][str(L)]={
                "reference_recall":ref["recall"],
                "same_L_recall_delta_pp":r["recall"]-ref["recall"],
                "same_L_extra_reads":r["reads"]-ref["reads"],
                "same_L_cpu_delta_us":r["cpu_us"]-ref["cpu_us"],
                "matched_recall":aligned,
                "matched_read_saving_vs_global":None if aligned is None else
                    100*(1-aligned["reads"]/ref["reads"]),
            }
    out={
        "status":"single-stream exploratory macro grouping, NOT replicated or equal actual RSS",
        "source":"one unchanged 16K-512 Hint-IVF, same train-only macro cell grouping",
        "queries":"PubMed/MedCPT 5K earlier training, 4K causal warm-up +1K measured",
        "hardware":"same pinned DiskANN and SSD, K=10 beam=8",
        "memory":"both have 2048 winner-ID slots and unchanged 16K static router; macro map adds 1040 bytes plus regional Vec and timestamps",
        "configs":list(MODES),
        "widths":list(LS),
        "original_ivf_sha256":original_ivf_hash,
        "macro_manifest":mm,
        "results":results,"stats":diag,
        "comparisons_vs_global2048":comparisons,
    }
    save(a.out/"macroregion-diagnostic.json",out)
    print("NAVHINTS_MACROREGION_DIAGNOSTIC_COMPLETE",flush=True)
    for L in LS:
        print(f"L={L} ref={reference[L]['recall']:.2f}%/{reference[L]['reads']:.3f} reads/{reference[L]['cpu_us']:.2f} CPUus",flush=True)
        for name,*_ in MODES:
            m=comparisons[name][str(L)]
            print(f"  {name:22} recall_delta={m['same_L_recall_delta_pp']:+.3f} "
                  f"reads_delta={m['same_L_extra_reads']:+.3f} "
                  f"CPU_delta_us={m['same_L_cpu_delta_us']:+.1f}",flush=True)

if __name__=="__main__":
    main()
